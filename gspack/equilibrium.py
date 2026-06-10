"""
gspack.equilibrium — CPU/GPU compatible Grad-Shafranov equilibrium.

GPU compatibility rules enforced here:
  - self.R, self.Z, self.plasma_psi  → always plain NumPy
    (SciPy RectBivariateSpline, find_critical, core_mask all require NumPy)
  - coil.psi() may return CuPy in GPU mode → always wrapped with to_numpy()
  - self.psi() always returns NumPy
  - free_boundary_hagenow receives to_backend() arrays and returns via to_numpy()
  - All β, li, q diagnostics operate purely in NumPy
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_erosion

from .backend  import MU0, to_numpy, to_backend
from .greens   import make_solver, make_masked_solver
from .boundary import (free_boundary_hagenow, fixed_boundary_solve,
                       dshape_lcfs, mask_inside_lcfs, initial_psi_lcfs,
                       greens_volume_psi)
from .critical import find_critical, core_mask


class Equilibrium:
    """
    Free-boundary Grad-Shafranov equilibrium.

    Parameters
    ----------
    tokamak  : Machine
    Rmin, Rmax, Zmin, Zmax : domain [m]
    nx, ny   : grid (use 2^n+1 for Romberg)
    order    : FDM order 2 (default) or 4
    method   : 'auto', 'lu', or 'amg'
    check_limited : detect limiter boundary
    """

    def __init__(self, tokamak,
                 Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                 nx=65, ny=65, order=2, method='auto',
                 check_limited=False):

        self.tokamak       = tokamak
        self.Rmin, self.Rmax = float(Rmin), float(Rmax)
        self.Zmin, self.Zmax = float(Zmin), float(Zmax)
        self.nx, self.ny   = int(nx), int(ny)
        self.check_limited = check_limited
        self.is_limited    = False

        # Grid — always NumPy (SciPy requirement)
        R1d = np.linspace(Rmin, Rmax, nx)
        Z1d = np.linspace(Zmin, Zmax, ny)
        self.R, self.Z = np.meshgrid(R1d, Z1d, indexing='ij')
        self.dR = float(R1d[1] - R1d[0])
        self.dZ = float(Z1d[1] - Z1d[0])

        # Solver (CPU; internally converts rhs via to_numpy/to_backend)
        self._solver = make_solver(Rmin, Rmax, Zmin, Zmax, nx, ny,
                                   order=order, method=method)

        # Gaussian initial plasma_psi — always NumPy
        xx, yy = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny), indexing='ij')
        plasma_psi = np.exp(-((xx-0.5)**2 + (yy-0.5)**2) / 0.4**2)
        plasma_psi[[0,-1],:] = 0.0
        plasma_psi[:,[0,-1]] = 0.0
        self.plasma_psi = plasma_psi   # float64 NumPy always

        self._profiles = None
        self._Jtor     = np.zeros((nx, ny))
        self._opoints  = []
        self._xpoints  = []
        self.psi_axis  = float(plasma_psi.max())
        self.psi_bndry = 0.0

        self._update_boundary_psi()

    # ── psi access ───────────────────────────────────────────────────────

    def psi(self):
        """Total ψ = plasma_psi + coil contributions.  Always NumPy."""
        result = self.plasma_psi.copy()
        for _, coil in self.tokamak.coils:
            # coil.psi() may be CuPy in GPU mode → to_numpy() is safe no-op on NumPy
            result += to_numpy(coil.psi(self.R, self.Z))
        return result

    def psiN(self):
        p = self.psi()
        d = self.psi_bndry - self.psi_axis
        if abs(d) < 1e-30:
            return np.zeros_like(p)
        return (p - self.psi_axis) / d

    def psiRZ(self, R, Z):
        f = RectBivariateSpline(self.R[:,0], self.Z[0,:], self.psi())
        return float(f(float(R), float(Z)).flat[0])

    # ── boundary update ──────────────────────────────────────────────────

    def _update_boundary_psi(self, psi=None):
        if psi is None:
            psi = self.psi()
        psi = to_numpy(psi)   # guarantee NumPy for find_critical / splines

        opt, xpt = find_critical(self.R, self.Z, psi)

        if not opt:
            opt_p, xpt_p = find_critical(self.R, self.Z, self.plasma_psi)
            if opt_p:
                f = RectBivariateSpline(self.R[:,0], self.Z[0,:], psi)
                Ro, Zo = opt_p[0][0], opt_p[0][1]
                opt = [(Ro, Zo, float(f(Ro, Zo).flat[0]))]
                xpt = xpt or xpt_p

        if not opt:
            idx = np.unravel_index(self.plasma_psi.argmax(), self.plasma_psi.shape)
            Ro = float(self.R[idx]); Zo = float(self.Z[idx])
            f  = RectBivariateSpline(self.R[:,0], self.Z[0,:], psi)
            opt = [(Ro, Zo, float(f(Ro, Zo).flat[0]))]

        self._opoints  = opt
        self._xpoints  = xpt
        self.psi_axis  = float(opt[0][2])

        if xpt:
            vals = [x[2] for x in xpt]
            pax  = self.psi_axis
            if pax > float(np.mean(vals)):
                cands = [v for v in vals if v < pax]
                self.psi_bndry = float(max(cands) if cands else vals[0])
            else:
                cands = [v for v in vals if v > pax]
                self.psi_bndry = float(min(cands) if cands else vals[0])
        else:
            bv = np.concatenate([psi[0,:], psi[-1,:], psi[:,0], psi[:,-1]])
            self.psi_bndry = float(np.mean(bv))

        if self.check_limited and self.tokamak.wall is not None and xpt:
            self._check_limiter(psi)

    def _check_limiter(self, psi):
        w   = self.tokamak.wall
        f   = RectBivariateSpline(self.R[:,0], self.Z[0,:], psi)
        pv  = [float(f(float(r), float(z)).flat[0]) for r,z in zip(w.R, w.Z)]
        plim = max(pv)
        if plim > self.psi_bndry:
            self.psi_bndry = float(plim); self.is_limited = True
        else:
            self.is_limited = False

    # ── magnetic field (all NumPy output) ────────────────────────────────

    def _psi_spline(self):
        return RectBivariateSpline(self.R[:,0], self.Z[0,:], self.psi())

    def Br(self, R, Z):
        f  = self._psi_spline()
        Ra = np.asarray(R, dtype=float)
        Za = np.asarray(Z, dtype=float)
        return -f(Ra, Za, dy=1, grid=False) / (Ra + 1e-30)

    def Bz(self, R, Z):
        f  = self._psi_spline()
        Ra = np.asarray(R, dtype=float)
        Za = np.asarray(Z, dtype=float)
        return f(Ra, Za, dx=1, grid=False) / (Ra + 1e-30)

    def Bpol(self, R=None, Z=None):
        if R is None: R = self.R
        if Z is None: Z = self.Z
        return np.sqrt(self.Br(R, Z)**2 + self.Bz(R, Z)**2)

    def Btor(self, R=None, Z=None):
        if R is None: R = self.R
        if Z is None: Z = self.Z
        Ra = np.asarray(R, dtype=float)
        Za = np.asarray(Z, dtype=float)
        if self._profiles is not None:
            f_psi    = RectBivariateSpline(self.R[:,0], self.Z[0,:], self.psi())
            psi_RZ   = f_psi(Ra.ravel(), Za.ravel(), grid=False).reshape(Ra.shape)
            pN       = np.clip((psi_RZ - self.psi_axis) /
                               (self.psi_bndry - self.psi_axis + 1e-30), 0, 1)
            f_val    = np.vectorize(self._profiles.fpol)(pN)
        else:
            f_val = 1.0
        return f_val / (Ra + 1e-30)

    @property
    def Jtor(self):
        return self._Jtor

    # ── one Picard step ──────────────────────────────────────────────────

    def solve(self, profiles, psi=None, psi_bndry=None):
        self._profiles = profiles
        if psi is None: psi = self.psi()
        psi = to_numpy(psi)   # ensure NumPy for splines / find_critical

        self._update_boundary_psi(psi)
        if psi_bndry is not None:
            self.psi_bndry = float(psi_bndry)

        psi_ax = self.psi_axis
        psi_bn = self.psi_bndry

        if self._opoints and self._xpoints:
            mask = core_mask(self.R, self.Z, psi, self._opoints, self._xpoints, psi_bn)
        else:
            psiN = (psi - psi_ax) / (psi_bn - psi_ax + 1e-30)
            mask = np.where((psiN >= 0) & (psiN <= 1), 1.0, 0.0)

        # Jtor — always NumPy
        Jtor = profiles.Jtor(self.R, self.Z, psi, psi_ax, psi_bn, mask=mask)
        self._Jtor = np.asarray(Jtor, dtype=float)

        # Free-boundary solve: convert to backend → solve → back to NumPy
        new_plasma_psi = to_numpy(
            free_boundary_hagenow(
                to_backend(self.R), to_backend(self.Z),
                to_backend(self._Jtor), self._solver
            )
        )
        self.plasma_psi = new_plasma_psi.astype(np.float64)
        self._update_boundary_psi()

    # ── diagnostics ──────────────────────────────────────────────────────

    def plasmaCurrent(self):
        return float(np.sum(self._Jtor) * self.dR * self.dZ)

    def magneticAxis(self):
        if self._opoints: return self._opoints[0]
        return (self.Rmin + (self.Rmax-self.Rmin)*0.5, 0.0, self.psi_axis)

    def separatrix(self, npoints=360):
        from .separatrix import find_separatrix
        return find_separatrix(self, ntheta=npoints)

    def innerOuterSeparatrix(self, Z=0.0):
        f    = RectBivariateSpline(self.R[:,0], self.Z[0,:], self.psi())
        mid  = f(self.R[:,0], float(Z))[:,0]
        R1d  = self.R[:,0]
        sign = mid - self.psi_bndry
        crossings = []
        for i in range(len(sign)-1):
            if sign[i]*sign[i+1] <= 0:
                frac = sign[i]/(sign[i]-sign[i+1]+1e-30)
                crossings.append(float(R1d[i] + frac*(R1d[i+1]-R1d[i])))
        return (min(crossings), max(crossings)) if len(crossings)>=2 else (self.Rmin, self.Rmax)

    def geometricAxis(self):
        sep = self.separatrix()
        return np.array([0.5*(sep[:,0].max()+sep[:,0].min()),
                         0.5*(sep[:,1].max()+sep[:,1].min())])

    def shafranovShift(self):
        mag = self.magneticAxis(); geo = self.geometricAxis()
        return np.array([mag[0]-geo[0], mag[1]-geo[1]])

    def minorRadius(self):
        Ri, Ro = self.innerOuterSeparatrix(); return 0.5*(Ro-Ri)

    def _shape_params(self):
        sep = self.separatrix(npoints=720)
        Rs, Zs = sep[:,0], sep[:,1]
        Rg = 0.5*(Rs.max()+Rs.min()); a = 0.5*(Rs.max()-Rs.min())
        b  = 0.5*(Zs.max()-Zs.min()); kappa = b/(a+1e-30)
        delta = 0.5*((Rg-Rs[int(np.argmax(Zs))])+(Rg-Rs[int(np.argmin(Zs))])) / (a+1e-30)
        return kappa, delta, Rg, 0.5*(Zs.max()+Zs.min()), a

    def elongation(self):    return self._shape_params()[0]
    def triangularity(self): return self._shape_params()[1]
    def aspectRatio(self):
        p = self._shape_params(); return p[2]/(p[4]+1e-30)

    def _masks(self):
        pN = self.psiN()
        return np.where((pN>=0)&(pN<=1),1.0,0.0), pN

    def poloidalBeta(self):
        if self._profiles is None: return 0.0
        mask, pN = self._masks()
        p2d  = np.vectorize(self._profiles.pressure)(np.clip(pN,0,1)) * mask
        Bp2  = (self.Br(self.R,self.Z)**2 + self.Bz(self.R,self.Z)**2) * mask
        return (2*MU0*float(np.sum(p2d*self.R))*self.dR*self.dZ /
                (float(np.sum(Bp2*self.R))*self.dR*self.dZ + 1e-30))

    def toroidalBeta(self):
        if self._profiles is None: return 0.0
        mask, pN = self._masks()
        p2d   = np.vectorize(self._profiles.pressure)(np.clip(pN,0,1)) * mask
        vol   = float(np.sum(mask*self.R)) * self.dR*self.dZ * 2*np.pi
        p_avg = float(np.sum(p2d*self.R)) * self.dR*self.dZ * 2*np.pi / (vol+1e-30)
        Rg    = self._shape_params()[2]
        Bt0   = self._profiles.fvac()/(Rg+1e-30)
        return 2*MU0*p_avg/(Bt0**2+1e-30)

    def totalBeta(self): return self.toroidalBeta()

    def betaN(self):
        bt = self.toroidalBeta(); a = self.minorRadius()
        Ip = abs(self.plasmaCurrent())
        Rg = self._shape_params()[2]
        Bt0 = (self._profiles.fvac()/(Rg+1e-30) if self._profiles else 1.0)
        return bt / (Ip/(a*Bt0+1e-30)*1e-6 + 1e-30)

    def internalInductance(self):
        mask, _ = self._masks()
        Bp2 = (self.Br(self.R,self.Z)**2+self.Bz(self.R,self.Z)**2)*mask
        Bp_int = float(np.sum(Bp2*self.R))*self.dR*self.dZ
        Ip  = abs(self.plasmaCurrent())+1e-30
        Rg  = self._shape_params()[2]
        return 2*Bp_int*Rg / ((MU0*Ip)**2/MU0)

    def plasmaVolume(self):
        mask, _ = self._masks()
        return float(2*np.pi*np.sum(mask*self.R)*self.dR*self.dZ)

    def pressure(self, pN):
        return self._profiles.pressure(pN) if self._profiles else 0.0

    def fpol(self, pN):
        return self._profiles.fpol(pN) if self._profiles else 1.0

    def q(self, psiN_arr):
        from .safety import find_safety
        return find_safety(self, np.asarray(psiN_arr, dtype=float))

    def printForces(self):
        print("  (Force calculation not yet implemented)")

    def printCurrents(self):
        self.tokamak.printCurrents()


# ═══════════════════════════════════════════════════════════════════════════
#  Fixed-boundary equilibrium (prescribed D-shaped LCFS, no coils)
# ═══════════════════════════════════════════════════════════════════════════

class FixedBoundaryEquilibrium:
    """
    Fixed-boundary Grad–Shafranov equilibrium with a prescribed D-shaped LCFS.

    The plasma boundary is defined by (R₀, a, κ, δ) — no external coils are
    used.  The GS equation is solved on a rectangular domain with Dirichlet
    boundary conditions computed from the plasma current via Green's function
    (fixed_boundary_solve).

    Current profile follows Jeon (2015) Eq. (5):
        J_φ = λ [β₀ R/R₀ + (1-β₀)R₀/R] (1 - ψ̂^{α_m})^{α_n}
    with λ, β₀ determined from (I_p, β_p) constraints (Eqs. 13a, 13b).

    Parameters
    ----------
    R0, a     : major and minor radius [m]
    kappa     : elongation
    delta     : triangularity
    Rmin, Rmax, Zmin, Zmax : computational domain [m]  (default: 0.2–1.8, -0.8–0.8)
    nx, ny    : grid — use 2ⁿ+1 for Romberg integration
    order     : FDM order 2 (default) or 4
    method    : 'auto', 'lu', or 'amg'
    check_limited : detect limiter contact (default False)
    """

    def __init__(self, R0=1.0, a=0.5, kappa=1.6, delta=0.3,
                 Rmin=0.2, Rmax=1.8, Zmin=-0.8, Zmax=0.8,
                 nx=65, ny=65, order=2, method='auto',
                 check_limited=False):

        self.R0, self.a     = float(R0), float(a)
        self.kappa          = float(kappa)
        self.delta          = float(delta)
        self.check_limited  = check_limited
        self.is_limited     = False

        # Computational domain
        self.Rmin, self.Rmax = float(Rmin), float(Rmax)
        self.Zmin, self.Zmax = float(Zmin), float(Zmax)
        self.nx, self.ny     = int(nx), int(ny)

        R1d = np.linspace(Rmin, Rmax, nx)
        Z1d = np.linspace(Zmin, Zmax, ny)
        self.R, self.Z = np.meshgrid(R1d, Z1d, indexing='ij')
        self.dR = float(R1d[1] - R1d[0])
        self.dZ = float(Z1d[1] - Z1d[0])

        # Solver — standard (unused for fixed-boundary solve)
        self._solver = make_solver(Rmin, Rmax, Zmin, Zmax, nx, ny,
                                   order=order, method=method)

        # D-shaped LCFS + mask
        self.psi_axis  = 1.0
        self.psi_bndry = 0.0
        self._update_lcfs()

        # D-shape-constrained solver:
        # Interior mask = plasma_mask eroded by 1 cell to ensure clean boundary
        # (prevents non-zero ψ from leaking to contour-adjacent grid points)
        self._interior_mask = binary_erosion(
            self.plasma_mask, structure=np.ones((3, 3)))
        self._dshape_solver = make_masked_solver(
            Rmin, Rmax, Zmin, Zmax, nx, ny, self._interior_mask,
            order=order, method=method)

        # Cerfon–Solovev-inspired initial guess (parabolic in ρ)
        self.plasma_psi = initial_psi_lcfs(
            self.R, self.Z, self.R_lcfs, self.Z_lcfs,
            psi_axis=self.psi_axis, psi_bndry=self.psi_bndry)

        self._profiles = None
        self._Jtor     = np.zeros((nx, ny))
        self._opoints  = []
        self._xpoints  = []

        self._update_boundary_psi()

    # ── LCFS helpers ──────────────────────────────────────────────────────

    def _update_lcfs(self):
        """(Re)generate D-shaped LCFS and inside/outside mask."""
        self.R_lcfs, self.Z_lcfs = dshape_lcfs(
            self.R0, self.a, self.kappa, self.delta, ntheta=360)
        self.plasma_mask = mask_inside_lcfs(
            self.R, self.Z, self.R_lcfs, self.Z_lcfs)

    # ── psi access ────────────────────────────────────────────────────────

    def psi(self):
        """Total ψ = plasma_psi (no coil contributions).  Always NumPy."""
        return to_numpy(self.plasma_psi.copy())

    def psiN(self):
        p = self.psi()
        d = self.psi_bndry - self.psi_axis
        if abs(d) < 1e-30:
            return np.zeros_like(p)
        return (p - self.psi_axis) / d

    def psiRZ(self, R, Z):
        f = RectBivariateSpline(self.R[:, 0], self.Z[0, :], self.psi())
        return float(f(float(R), float(Z)).flat[0])

    # ── boundary update ───────────────────────────────────────────────────

    def _update_boundary_psi(self, psi=None):
        """Update psi_axis, psi_bndry, find O/X points from ψ field."""
        if psi is None:
            psi = self.psi()
        psi = to_numpy(psi)

        opt, xpt = find_critical(self.R, self.Z, psi)

        # Fallback: use plasma_mask centroid
        if not opt:
            idx = np.unravel_index(
                (psi * self.plasma_mask).argmax(), psi.shape)
            Ro = float(self.R[idx]); Zo = float(self.Z[idx])
            f  = RectBivariateSpline(self.R[:, 0], self.Z[0, :], psi)
            opt = [(Ro, Zo, float(f(Ro, Zo).flat[0]))]

        self._opoints = opt
        self._xpoints = xpt
        self.psi_axis = float(opt[0][2])

        # Boundary: use the LCFS-masked edge values
        if xpt:
            vals = [x[2] for x in xpt]
            pax  = self.psi_axis
            if pax > float(np.mean(vals)):
                cands = [v for v in vals if v < pax]
                self.psi_bndry = float(max(cands) if cands else vals[0])
            else:
                cands = [v for v in vals if v > pax]
                self.psi_bndry = float(min(cands) if cands else vals[0])
        else:
            bv = np.concatenate([psi[0, :], psi[-1, :],
                                 psi[:, 0], psi[:, -1]])
            self.psi_bndry = float(np.mean(bv))

    # ── magnetic field ────────────────────────────────────────────────────

    def _psi_spline(self):
        return RectBivariateSpline(self.R[:, 0], self.Z[0, :], self.psi())

    def Br(self, R, Z):
        f  = self._psi_spline()
        Ra = np.asarray(R, dtype=float)
        Za = np.asarray(Z, dtype=float)
        return -f(Ra, Za, dy=1, grid=False) / (Ra + 1e-30)

    def Bz(self, R, Z):
        f  = self._psi_spline()
        Ra = np.asarray(R, dtype=float)
        Za = np.asarray(Z, dtype=float)
        return f(Ra, Za, dx=1, grid=False) / (Ra + 1e-30)

    def Bpol(self, R=None, Z=None):
        if R is None:
            R = self.R
        if Z is None:
            Z = self.Z
        return np.sqrt(self.Br(R, Z)**2 + self.Bz(R, Z)**2)

    def Btor(self, R=None, Z=None):
        if R is None:
            R = self.R
        if Z is None:
            Z = self.Z
        Ra = np.asarray(R, dtype=float)
        Za = np.asarray(Z, dtype=float)
        if self._profiles is not None:
            f_psi  = RectBivariateSpline(self.R[:, 0], self.Z[0, :], self.psi())
            psi_RZ = f_psi(Ra.ravel(), Za.ravel(),
                           grid=False).reshape(Ra.shape)
            pN = np.clip((psi_RZ - self.psi_axis) /
                         (self.psi_bndry - self.psi_axis + 1e-30), 0, 1)
            f_val = np.vectorize(self._profiles.fpol)(pN)
        else:
            f_val = 1.0
        return f_val / (Ra + 1e-30)

    @property
    def Jtor(self):
        return self._Jtor

    # ── one Picard step ───────────────────────────────────────────────────

    def solve(self, profiles, psi=None, psi_bndry=None):
        """
        Single Picard iteration with fixed-boundary solve.

        For each iteration:
          1. Compute J_φ from profiles using current ψ
          2. Compute ψ_green = ∫∫ G·J_φ dS on the D-shape boundary (free-space)
          3. Set ψ_bndry = mean(ψ_green on D-shape) — the LCFS value
          4. Solve GS inside D-shape with ψ = ψ_bndry Dirichlet BC

        After convergence, the Green integral from the converged Jtor gives
        ψ ≈ ψ_bndry on the D-shape contour — this is the self-consistent
        fixed-boundary solution.  ψ on the D-shape is a flux surface.

        Parameters
        ----------
        profiles : profile object with Jtor() method (e.g. ConstrainBetapIp)
        psi      : current total ψ (if None, uses self.psi())
        psi_bndry: override (optional); if None, computed from Green integral
        """
        self._profiles = profiles
        if psi is None:
            psi = self.psi()
        psi = to_numpy(psi)

        # ── Update psi_axis from current ψ (preserve psi_bndry) ───────────
        opt, xpt = find_critical(self.R, self.Z, psi)
        if opt:
            self._opoints = opt
            self._xpoints = xpt
            self.psi_axis = float(opt[0][2])
        else:
            self.psi_axis = float(psi.max())
        # Keep existing psi_bndry (from previous Green integral or initial)
        if psi_bndry is not None:
            self.psi_bndry = float(psi_bndry)

        psi_ax = self.psi_axis
        psi_bn = self.psi_bndry
        mask = self.plasma_mask.astype(float)

        # ── 1. Compute Jtor from profiles (using current ψ_bndry) ─────────
        Jtor = profiles.Jtor(self.R, self.Z, psi,
                             psi_ax, psi_bn, mask=mask)
        self._Jtor = np.asarray(Jtor, dtype=float)

        # ── 2. Compute new ψ_bndry from Green volume integral ─────────────
        if psi_bndry is None:
            idx_i, idx_j = np.where(self.plasma_mask)

            # Cache Green matrix (D-shape × source points, unchanged)
            if not hasattr(self, '_G_dshape') or self._G_dshape.shape[1] != len(idx_i):
                from .boundary import _green_matrix_np
                self._G_dshape = _green_matrix_np(
                    np.asarray(self.R_lcfs).ravel(),
                    np.asarray(self.Z_lcfs).ravel(),
                    self.R[idx_i, idx_j], self.Z[idx_i, idx_j])

            psi_green_bc = (
                self._G_dshape @ (self._Jtor[idx_i, idx_j] * self.dR * self.dZ))

            self.psi_bndry = float(psi_green_bc.mean())

        psi_bn = self.psi_bndry

        # ── 3. Solve GS inside D-shape with ψ = ψ_bndry Dirichlet BC ──────
        # rhs[interior] = -μ₀ R Jtor, rhs[boundary] = ψ_bndry
        rhs = np.full_like(self.R, psi_bn, dtype=float)
        rhs[self._interior_mask] = (
            -MU0 * self.R[self._interior_mask] * self._Jtor[self._interior_mask])

        new_plasma_psi = to_numpy(
            self._dshape_solver(to_backend(rhs))
        )
        self.plasma_psi = new_plasma_psi.astype(np.float64)

        # ── 4. Update psi_axis ────────────────────────────────────────────
        opt, xpt = find_critical(self.R, self.Z, self.plasma_psi)
        if opt:
            self._opoints = opt
            self._xpoints = xpt
            self.psi_axis = float(opt[0][2])
        else:
            self.psi_axis = float(self.plasma_psi.max())

    # ── external-domain ψ via Green integral ──────────────────────────────

    def psi_on_grid(self, Rmin, Rmax, Zmin, Zmax, nx, ny, order=2, method='lu'):
        """
        Compute ψ on a larger vessel-scale grid via Green's function
        volume integral.

        ψ(R,Z) = ∫∫ G(R,Z; R',Z') · J_φ(R',Z') dR' dZ'

        This is the exact free-space poloidal flux from the converged
        plasma current distribution.  It satisfies:
          • Δ*ψ = -μ₀ R Jφ   inside the plasma
          • Δ*ψ = 0           in vacuum
          • ψ → 0             at infinity

        The Green integral gives a single, continuous ψ field everywhere.
        Mixed with FDM interior interpolation → avoid discontinuity at
        the D-shape boundary.  On the D-shape, ψ varies (not constant)
        because the Green kernel does not enforce a Dirichlet BC — the
        constant-ψ LCFS is only exactly enforced in the FDM solve used
        for internal diagnostics.

        Parameters
        ----------
        Rmin, Rmax, Zmin, Zmax : domain bounds [m]
        nx, ny : grid size for the large domain
        order, method : unused (API compatibility)

        Returns
        -------
        R2d, Z2d, psi : 2-D arrays — the coordinate mesh and ψ field
        """
        R1d = np.linspace(Rmin, Rmax, nx)
        Z1d = np.linspace(Zmin, Zmax, ny)
        R2d, Z2d = np.meshgrid(R1d, Z1d, indexing='ij')

        idx_i, idx_j = np.where(self.plasma_mask)
        psi = greens_volume_psi(
            R2d.ravel(), Z2d.ravel(),
            self.R[idx_i, idx_j], self.Z[idx_i, idx_j],
            self._Jtor[idx_i, idx_j],
            self.dR, self.dZ)

        return R2d, Z2d, psi.reshape(R2d.shape)

    # ── diagnostics ───────────────────────────────────────────────────────

    def plasmaCurrent(self):
        return float(np.sum(self._Jtor) * self.dR * self.dZ)

    def magneticAxis(self):
        if self._opoints:
            return self._opoints[0]
        return (self.Rmin + (self.Rmax - self.Rmin) * 0.5,
                0.0, self.psi_axis)

    def separatrix(self, npoints=360):
        """Return LCFS contour (the prescribed D-shape)."""
        return np.column_stack([self.R_lcfs, self.Z_lcfs])

    def innerOuterSeparatrix(self, Z=0.0):
        f    = RectBivariateSpline(self.R[:, 0], self.Z[0, :], self.psi())
        mid  = f(self.R[:, 0], float(Z))[:, 0]
        R1d  = self.R[:, 0]
        sign = mid - self.psi_bndry
        crossings = []
        for i in range(len(sign) - 1):
            if sign[i] * sign[i + 1] <= 0:
                frac = sign[i] / (sign[i] - sign[i + 1] + 1e-30)
                crossings.append(
                    float(R1d[i] + frac * (R1d[i + 1] - R1d[i])))
        return (min(crossings), max(crossings)) if len(crossings) >= 2 else (
            self.Rmin, self.Rmax)

    def geometricAxis(self):
        R_g = 0.5 * (self.R_lcfs.max() + self.R_lcfs.min())
        Z_g = 0.5 * (self.Z_lcfs.max() + self.Z_lcfs.min())
        return np.array([R_g, Z_g])

    def shafranovShift(self):
        mag = self.magneticAxis()
        geo = self.geometricAxis()
        return np.array([mag[0] - geo[0], mag[1] - geo[1]])

    def minorRadius(self):
        Ri, Ro = self.innerOuterSeparatrix()
        return 0.5 * (Ro - Ri)

    def _shape_params(self):
        Rs, Zs = self.R_lcfs, self.Z_lcfs
        Rg     = 0.5 * (Rs.max() + Rs.min())
        a      = 0.5 * (Rs.max() - Rs.min())
        b      = 0.5 * (Zs.max() - Zs.min())
        kappa  = b / (a + 1e-30)
        delta  = 0.5 * ((Rg - Rs[int(np.argmax(Zs))]) +
                        (Rg - Rs[int(np.argmin(Zs))])) / (a + 1e-30)
        return kappa, delta, Rg, 0.5 * (Zs.max() + Zs.min()), a

    def elongation(self):
        return self._shape_params()[0]

    def triangularity(self):
        return self._shape_params()[1]

    def aspectRatio(self):
        p = self._shape_params()
        return p[2] / (p[4] + 1e-30)

    def _masks(self):
        """Return (mask, psiN) where mask = 1 inside the prescribed D-shape LCFS."""
        pN   = self.psiN()
        mask = self.plasma_mask.astype(float)
        return mask, pN

    def poloidalBeta(self):
        if self._profiles is None:
            return 0.0
        mask, pN = self._masks()
        p2d = np.vectorize(
            self._profiles.pressure)(np.clip(pN, 0, 1)) * mask
        Bp2 = (self.Br(self.R, self.Z)**2 +
               self.Bz(self.R, self.Z)**2) * mask
        return (2 * MU0 * float(np.sum(p2d * self.R)) * self.dR * self.dZ /
                (float(np.sum(Bp2 * self.R)) * self.dR * self.dZ + 1e-30))

    def toroidalBeta(self):
        if self._profiles is None:
            return 0.0
        mask, pN = self._masks()
        p2d  = np.vectorize(
            self._profiles.pressure)(np.clip(pN, 0, 1)) * mask
        vol  = float(np.sum(mask * self.R)) * self.dR * self.dZ * 2 * np.pi
        p_avg = float(np.sum(p2d * self.R)) * self.dR * self.dZ * 2 * np.pi / (
            vol + 1e-30)
        Rg   = self._shape_params()[2]
        Bt0  = self._profiles.fvac() / (Rg + 1e-30)
        return 2 * MU0 * p_avg / (Bt0**2 + 1e-30)

    def totalBeta(self):
        return self.toroidalBeta()

    def betaN(self):
        bt = self.toroidalBeta()
        a  = self.minorRadius()
        Ip = abs(self.plasmaCurrent())
        Rg = self._shape_params()[2]
        Bt0 = (self._profiles.fvac() / (Rg + 1e-30)
               if self._profiles else 1.0)
        return bt / (Ip / (a * Bt0 + 1e-30) * 1e-6 + 1e-30)

    def internalInductance(self):
        mask, _ = self._masks()
        Bp2  = (self.Br(self.R, self.Z)**2 +
                self.Bz(self.R, self.Z)**2) * mask
        Bp_int = float(np.sum(Bp2 * self.R)) * self.dR * self.dZ
        Ip     = abs(self.plasmaCurrent()) + 1e-30
        Rg     = self._shape_params()[2]
        return 2 * Bp_int * Rg / ((MU0 * Ip)**2 / MU0)

    def plasmaVolume(self):
        mask, _ = self._masks()
        return float(2 * np.pi * np.sum(mask * self.R) * self.dR * self.dZ)

    def pressure(self, pN):
        return self._profiles.pressure(pN) if self._profiles else 0.0

    def fpol(self, pN):
        return self._profiles.fpol(pN) if self._profiles else 1.0

    def q(self, psiN_arr):
        from .safety import find_safety
        return find_safety(self, np.asarray(psiN_arr, dtype=float))

    def printForces(self):
        print("  (Force calculation not yet implemented)")

    def printCurrents(self):
        print("  (Fixed-boundary: no external coils)")
