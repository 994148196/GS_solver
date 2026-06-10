"""
gspack.boundary
===============
Free-boundary + fixed-boundary Grad-Shafranov solvers.

Provides:
  • free_boundary_hagenow  — von Hagenow free-boundary condition
  • fixed_boundary_solve   — fixed-boundary solve with prescribed LCFS shape
  • D-shaped LCFS utilities: dshape_lcfs, mask_inside_lcfs, initial_psi_lcfs

GPU transparency: all routines handle NumPy/CuPy arrays transparently.
"""

import numpy as np
try:
    _trapz = np.trapezoid   # NumPy ≥ 2.0
except AttributeError:
    _trapz = np.trapz       # NumPy < 2.0

from scipy.integrate import romb as _romb
from .backend import MU0, get_xp, to_numpy, to_backend, asarray
from .greens  import greens


# 4th-order one-sided FD coefficients for outward normal derivative
_COEFFS = [(0, 25.0/12), (1, -4.0), (2, 3.0), (3, -16.0/12), (4, 0.25)]


def _normal_deriv_left(psi, dR):
    return sum(w * psi[k,  :] for k, w in _COEFFS) / dR

def _normal_deriv_right(psi, dR):
    return sum(w * psi[-(1+k), :] for k, w in _COEFFS) / dR

def _normal_deriv_bottom(psi, dZ):
    return sum(w * psi[:, k]  for k, w in _COEFFS) / dZ

def _normal_deriv_top(psi, dZ):
    return sum(w * psi[:, -(1+k)] for k, w in _COEFFS) / dZ


def _green_matrix_np(R_obs, Z_obs, R_src, Z_src):
    """
    Build Green's matrix entirely in NumPy (for boundary integral).
    Shape: (n_obs, n_src).

    We always use NumPy here because scipy.integrate.trapezoid / romb
    require numpy arrays.
    """
    # Ensure numpy scalars/arrays
    R_obs = np.asarray(R_obs, dtype=float)
    Z_obs = np.asarray(Z_obs, dtype=float)
    R_src = np.asarray(R_src, dtype=float)
    Z_src = np.asarray(Z_src, dtype=float)

    R_o = R_obs[:, None]
    Z_o = Z_obs[:, None]
    R_s = R_src[None, :]
    Z_s = Z_src[None, :]

    # Compute greens on CPU regardless of global backend (boundary integral is CPU)
    from scipy.special import ellipk, ellipe
    k2 = 4.0 * R_o * R_s / ((R_o + R_s)**2 + (Z_o - Z_s)**2)
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)
    k  = np.sqrt(k2)
    return (MU0 / (2.0 * np.pi)) * np.sqrt(R_o * R_s) \
           * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)) / k


def free_boundary_hagenow(R, Z, Jtor, solver):
    """
    Von Hagenow free-boundary condition.

    Parameters  (may be numpy or cupy arrays)
    ----------
    R, Z   : 2-D meshgrid (nx, ny)
    Jtor   : 2-D toroidal current density
    solver : callable solve(rhs_2d) → psi_2d

    Returns
    -------
    psi_plasma : 2-D array on active backend device
    """
    # Work in numpy for all boundary integrals (scipy requirement)
    R_np   = to_numpy(R)
    Z_np   = to_numpy(Z)
    Jt_np  = to_numpy(Jtor)

    nx, ny = R_np.shape
    R1d    = R_np[:, 0]
    Z1d    = Z_np[0, :]
    dR     = float(R1d[1] - R1d[0])
    dZ     = float(Z1d[1] - Z1d[0])

    # ── Step 1: fixed-boundary solve ─────────────────────────────────────
    rhs = np.zeros((nx, ny))
    rhs[1:-1, 1:-1] = -MU0 * R_np[1:-1, 1:-1] * Jt_np[1:-1, 1:-1]
    psi_fixed = to_numpy(solver(to_backend(rhs)))

    # ── Step 2: 4th-order normal derivatives ─────────────────────────────
    dUdn_L = _normal_deriv_left  (psi_fixed, dR)
    dUdn_R = _normal_deriv_right (psi_fixed, dR)
    dUdn_B = _normal_deriv_bottom(psi_fixed, dZ)
    dUdn_T = _normal_deriv_top   (psi_fixed, dZ)

    # Corner corrections
    dd = np.sqrt(dR**2 + dZ**2)
    dUdn_L[0]  = dUdn_B[0]  = sum(w*psi_fixed[k, k]          for k,w in _COEFFS)/dd
    dUdn_L[-1] = dUdn_T[0]  = sum(w*psi_fixed[k, -(1+k)]     for k,w in _COEFFS)/dd
    dUdn_R[0]  = dUdn_B[-1] = sum(w*psi_fixed[-(1+k), k]     for k,w in _COEFFS)/dd
    dUdn_R[-1] = dUdn_T[-1] = sum(w*psi_fixed[-(1+k),-(1+k)] for k,w in _COEFFS)/dd

    # ── Step 3: vectorised boundary contour integral ──────────────────────
    eps = 1e-2

    R_obs_B  = R1d;                       Z_obs_B  = np.full(nx, Z1d[0]  - eps)
    R_obs_T  = R1d;                       Z_obs_T  = np.full(nx, Z1d[-1] + eps)
    R_obs_Lj = np.full(ny-2, R1d[0] -eps); Z_obs_Lj = Z1d[1:-1]
    R_obs_Rj = np.full(ny-2, R1d[-1]+eps); Z_obs_Rj = Z1d[1:-1]

    R_obs_all = np.concatenate([R_obs_B, R_obs_T, R_obs_Lj, R_obs_Rj])
    Z_obs_all = np.concatenate([Z_obs_B, Z_obs_T, Z_obs_Lj, Z_obs_Rj])

    # Outward-normal derivatives:
    #   Left:  n̂ = -R̂ → ∂ψ/∂n = -∂ψ/∂R = -dUdn_L
    #   Right: n̂ = +R̂ → ∂ψ/∂n = +∂ψ/∂R = +dUdn_R
    #   Bottom: n̂ = -ẑ → ∂ψ/∂n = -∂ψ/∂Z = -dUdn_B
    #   Top:   n̂ = +ẑ → ∂ψ/∂n = +∂ψ/∂Z = +dUdn_T
    wL = (-dUdn_L / R1d[0]) * dZ
    wR = ( dUdn_R / R1d[-1]) * dZ
    wB = (-dUdn_B / R1d) * dR
    wT = ( dUdn_T / R1d) * dR

    def _int(R_o, Z_o, R_s, Z_s, w):
        G = _green_matrix_np(R_o, Z_o, R_s, Z_s)
        return _trapz(G * w[np.newaxis, :], dx=1.0, axis=1)

    psi_obs = (
        _int(R_obs_all, Z_obs_all, np.full(len(R1d), R1d[0]),  Z1d, wL) +
        _int(R_obs_all, Z_obs_all, np.full(len(R1d), R1d[-1]), Z1d, wR) +
        _int(R_obs_all, Z_obs_all, R1d, np.full(len(Z1d), Z1d[0]),  wB) +
        _int(R_obs_all, Z_obs_all, R1d, np.full(len(Z1d), Z1d[-1]), wT)
    )

    psi_bndry = np.zeros((nx, ny))
    psi_bndry[:, 0]     = psi_obs[:nx]
    psi_bndry[:, -1]    = psi_obs[nx:2*nx]
    psi_bndry[0,  1:-1] = psi_obs[2*nx         : 2*nx+(ny-2)]
    psi_bndry[-1, 1:-1] = psi_obs[2*nx+(ny-2)  : 2*nx+2*(ny-2)]

    # ── Step 4: re-solve with free boundary ───────────────────────────────
    rhs2 = np.zeros((nx, ny))
    rhs2[1:-1, 1:-1] = -MU0 * R_np[1:-1, 1:-1] * Jt_np[1:-1, 1:-1]
    rhs2[0,  :] = psi_bndry[0,  :]
    rhs2[-1, :] = psi_bndry[-1, :]
    rhs2[:,  0] = psi_bndry[:,  0]
    rhs2[:, -1] = psi_bndry[:, -1]

    return solver(to_backend(rhs2))


# ═══════════════════════════════════════════════════════════════════════════
#  Green's function: volume integral utilities
# ═══════════════════════════════════════════════════════════════════════════

def greens_volume_psi(R_obs, Z_obs, R_src, Z_src, Jtor_src, dR, dZ):
    """
    Compute ψ at observation points via Green's function volume integral.

    ψ(R,Z) = ∫∫ G(R,Z; R',Z') · J_φ(R',Z') dR' dZ'

    where G is the poloidal flux at (R,Z) from a unit toroidal current
    filament at (R',Z').  The GS equation is Δ*ψ = -μ₀ R J_φ, and G
    satisfies Δ*G = -μ₀ R δ(R-R')δ(Z-Z'), so the convolution gives
    the free-space solution.

    Suitable for computing ψ anywhere from a known current distribution,
    including the external vacuum region (Laplace equation).

    Parameters
    ----------
    R_obs, Z_obs : 1-D arrays — observation points
    R_src, Z_src : 1-D arrays — source (plasma) points
    Jtor_src     : 1-D array  — J_φ at source points [A/m²]
    dR, dZ       : float — grid spacing [m]

    Returns
    -------
    psi_obs : 1-D array, len(R_obs) — poloidal flux at observation points
    """
    R_obs = np.asarray(R_obs, dtype=float).ravel()
    Z_obs = np.asarray(Z_obs, dtype=float).ravel()
    R_src = np.asarray(R_src, dtype=float).ravel()
    Z_src = np.asarray(Z_src, dtype=float).ravel()
    Jtor  = np.asarray(Jtor_src, dtype=float).ravel()

    G = _green_matrix_np(R_obs, Z_obs, R_src, Z_src)
    return G @ (Jtor * dR * dZ)


# ═══════════════════════════════════════════════════════════════════════════
#  Fixed-boundary utilities and solver
# ═══════════════════════════════════════════════════════════════════════════

def dshape_lcfs(R0, a, kappa, delta, ntheta=360):
    """
    Generate D-shaped LCFS (last-closed flux surface) coordinates
    using the standard parameterisation (Cerfon–Solovev style).

    Parameters
    ----------
    R0     : float — major radius [m]
    a      : float — minor radius [m]
    kappa  : float — elongation
    delta  : float — triangularity
    ntheta : int   — number of poloidal points

    Returns
    -------
    R_lcfs, Z_lcfs : 1-D ndarrays of LCFS contour points
    """
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    R_lcfs = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z_lcfs = kappa * a * np.sin(theta)
    return R_lcfs, Z_lcfs


def mask_inside_lcfs(R_grid, Z_grid, R_lcfs, Z_lcfs):
    """
    Return boolean mask: True for grid points inside the LCFS contour.

    Uses matplotlib.path (if available) for efficient point-in-polygon
    test; falls back to ray-casting algorithm.

    Parameters
    ----------
    R_grid, Z_grid : 2-D arrays (nx, ny) from np.meshgrid(..., indexing='ij')
    R_lcfs, Z_lcfs : 1-D arrays defining the closed LCFS contour

    Returns
    -------
    mask : (nx, ny) bool ndarray
    """
    points = np.column_stack([R_grid.ravel(), Z_grid.ravel()])

    try:
        from matplotlib.path import Path
        contour = np.column_stack([R_lcfs, Z_lcfs])
        mask = Path(contour).contains_points(points).reshape(R_grid.shape)
        return mask
    except ImportError:
        # Ray-casting fallback — fully vectorised
        n = len(R_lcfs)
        R_flat, Z_flat = R_grid.ravel(), Z_grid.ravel()
        inside = np.zeros(len(R_flat), dtype=bool)

        for idx in range(n):
            jdx = (idx + 1) % n
            Ri, Zi = R_lcfs[idx], Z_lcfs[idx]
            Rj, Zj = R_lcfs[jdx], Z_lcfs[jdx]

            # Mask: y-range crossing and x-condition
            y_cond = ((Zi > Z_flat) != (Zj > Z_flat))
            x_cond = (R_flat < (Rj - Ri) * (Z_flat - Zi) / (Zj - Zi + 1e-30) + Ri)
            inside[y_cond & x_cond] = ~inside[y_cond & x_cond]

        return inside.reshape(R_grid.shape)


def initial_psi_lcfs(R_grid, Z_grid, R_lcfs, Z_lcfs,
                     psi_axis=1.0, psi_bndry=0.0):
    """
    Generate an initial ψ guess using normalised flux coordinate ρ.

    The magnetic axis is assumed at the geometric centre of the LCFS.
    For each grid point inside the LCFS:
        ρ² = d²_axis / d²_LCFS(θ)   (ratio of distances)
        ψ(ρ) = ψ_bndry + (ψ_axis - ψ_bndry) * (1 - ρ²)

    This gives a parabolic ψ profile aligned with the D-shaped boundary,
    analogous to the Cerfon–Solovev (ρ, θ) coordinate system.

    Parameters
    ----------
    R_grid, Z_grid : 2-D arrays (nx, ny)
    R_lcfs, Z_lcfs : 1-D LCFS contour arrays
    psi_axis, psi_bndry : float — flux values at axis and boundary

    Returns
    -------
    psi_init : (nx, ny) ndarray — initial ψ guess
    """
    nx, ny = R_grid.shape
    R0 = float(np.mean(R_lcfs))
    Z0 = float(np.mean(Z_lcfs))

    # Build LCFS radial lookup table for all angles
    theta_lcfs = np.arctan2(Z_lcfs - Z0, R_lcfs - R0) % (2 * np.pi)
    R_lcfs_aligned = np.interp(theta_lcfs, theta_lcfs, R_lcfs)  # sort by angle
    Z_lcfs_aligned = np.interp(theta_lcfs, theta_lcfs, Z_lcfs)

    psi_init = np.full((nx, ny), psi_bndry, dtype=float)

    for i in range(nx):
        for j in range(ny):
            r, z = R_grid[i, j], Z_grid[i, j]
            dx = r - R0
            dz = z - Z0
            d_sq = dx**2 + dz**2
            if d_sq < 1e-30:
                psi_init[i, j] = psi_axis
                continue

            # Find angular position and LCFS distance at same angle
            theta = np.arctan2(dz, dx) % (2 * np.pi)
            # Linear interpolation to find LCFS intersection distance
            R_l = float(np.interp(theta, theta_lcfs, R_lcfs_aligned))
            Z_l = float(np.interp(theta, theta_lcfs, Z_lcfs_aligned))
            d_lcfs_sq = (R_l - R0)**2 + (Z_l - Z0)**2

            if d_sq <= d_lcfs_sq * 1.001:
                rho2 = d_sq / (d_lcfs_sq + 1e-30)
                psi_init[i, j] = psi_bndry + (psi_axis - psi_bndry) * (1.0 - rho2)

    return psi_init


def fixed_boundary_solve(R, Z, Jtor, solver):
    """
    Fixed-boundary Grad–Shafranov solve.

    Uses Green's theorem to compute ψ on the rectangular domain boundary
    from the plasma current distribution, then solves
        Δ*ψ = -μ₀ R Jtor
    with the resulting Dirichlet boundary conditions.

    This is the first half of the von Hagenow method (fixed-boundary solve
    + boundary integral), without the free-boundary iteration.

    Parameters
    ----------
    R, Z   : 2-D meshgrid arrays (nx, ny) — CPU or GPU
    Jtor   : 2-D toroidal current density (zero outside LCFS)  [A/m²]
    solver : callable solve(rhs_2d) → psi_2d

    Returns
    -------
    psi : 2-D array on active backend
    """
    R_np  = to_numpy(R)
    Z_np  = to_numpy(Z)
    Jt_np = to_numpy(Jtor)

    nx, ny = R_np.shape
    R1d    = R_np[:, 0]
    Z1d    = Z_np[0, :]
    dR     = float(R1d[1] - R1d[0])
    dZ     = float(Z1d[1] - Z1d[0])

    # ── Step 1: fixed-boundary solve (zero Dirichlet BC) ────────────────
    rhs = np.zeros((nx, ny))
    rhs[1:-1, 1:-1] = -MU0 * R_np[1:-1, 1:-1] * Jt_np[1:-1, 1:-1]
    psi_fixed = to_numpy(solver(to_backend(rhs)))

    # ── Step 2: 4th-order normal derivatives ─────────────────────────────
    dUdn_L = _normal_deriv_left  (psi_fixed, dR)
    dUdn_R = _normal_deriv_right (psi_fixed, dR)
    dUdn_B = _normal_deriv_bottom(psi_fixed, dZ)
    dUdn_T = _normal_deriv_top   (psi_fixed, dZ)

    # Corner corrections
    dd = np.sqrt(dR**2 + dZ**2)
    dUdn_L[0]  = dUdn_B[0]  = sum(w * psi_fixed[k, k]          for k, w in _COEFFS) / dd
    dUdn_L[-1] = dUdn_T[0]  = sum(w * psi_fixed[k, -(1 + k)]   for k, w in _COEFFS) / dd
    dUdn_R[0]  = dUdn_B[-1] = sum(w * psi_fixed[-(1 + k), k]   for k, w in _COEFFS) / dd
    dUdn_R[-1] = dUdn_T[-1] = sum(w * psi_fixed[-(1 + k), -(1 + k)] for k, w in _COEFFS) / dd

    # ── Step 3: boundary contour integral via Green's function ──────────
    eps = 1e-2

    R_obs_B  = R1d
    Z_obs_B  = np.full(nx, Z1d[0] - eps)
    R_obs_T  = R1d
    Z_obs_T  = np.full(nx, Z1d[-1] + eps)
    R_obs_Lj = np.full(ny - 2, R1d[0] - eps)
    Z_obs_Lj = Z1d[1:-1]
    R_obs_Rj = np.full(ny - 2, R1d[-1] + eps)
    Z_obs_Rj = Z1d[1:-1]

    R_obs_all = np.concatenate([R_obs_B, R_obs_T, R_obs_Lj, R_obs_Rj])
    Z_obs_all = np.concatenate([Z_obs_B, Z_obs_T, Z_obs_Lj, Z_obs_Rj])

    # Outward-normal derivatives:
    #   Left:  n̂ = -R̂ → ∂ψ/∂n = -∂ψ/∂R = -dUdn_L
    #   Right: n̂ = +R̂ → ∂ψ/∂n = +∂ψ/∂R = +dUdn_R
    #   Bottom: n̂ = -ẑ → ∂ψ/∂n = -∂ψ/∂Z = -dUdn_B
    #   Top:   n̂ = +ẑ → ∂ψ/∂n = +∂ψ/∂Z = +dUdn_T
    wL = (-dUdn_L / R1d[0]) * dZ
    wR = ( dUdn_R / R1d[-1]) * dZ
    wB = (-dUdn_B / R1d) * dR
    wT = ( dUdn_T / R1d) * dR

    def _int(R_o, Z_o, R_s, Z_s, w):
        G = _green_matrix_np(R_o, Z_o, R_s, Z_s)
        return _trapz(G * w[np.newaxis, :], dx=1.0, axis=1)

    psi_obs = (
        _int(R_obs_all, Z_obs_all, np.full(len(R1d), R1d[0]), Z1d, wL)
        + _int(R_obs_all, Z_obs_all, np.full(len(R1d), R1d[-1]), Z1d, wR)
        + _int(R_obs_all, Z_obs_all, R1d, np.full(len(Z1d), Z1d[0]), wB)
        + _int(R_obs_all, Z_obs_all, R1d, np.full(len(Z1d), Z1d[-1]), wT)
    )

    # Map observation points → rectangular boundary
    psi_bndry = np.zeros((nx, ny))
    psi_bndry[:, 0]      = psi_obs[:nx]
    psi_bndry[:, -1]     = psi_obs[nx:2 * nx]
    psi_bndry[0, 1:-1]   = psi_obs[2 * nx: 2 * nx + (ny - 2)]
    psi_bndry[-1, 1:-1]  = psi_obs[2 * nx + (ny - 2): 2 * nx + 2 * (ny - 2)]

    # ── Step 4: re-solve with correct boundary values ────────────────────
    rhs2 = np.zeros((nx, ny))
    rhs2[1:-1, 1:-1] = -MU0 * R_np[1:-1, 1:-1] * Jt_np[1:-1, 1:-1]
    rhs2[0, :]  = psi_bndry[0, :]
    rhs2[-1, :] = psi_bndry[-1, :]
    rhs2[:, 0]  = psi_bndry[:, 0]
    rhs2[:, -1] = psi_bndry[:, -1]

    return solver(to_backend(rhs2))
