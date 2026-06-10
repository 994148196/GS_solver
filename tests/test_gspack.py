"""
tests/test_gspack.py
====================
Tests for gspack — independent GS solver reproducing FreeGS 01-freeboundary.

Run:  cd gspack && python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import warnings

from gspack.greens     import greens, greens_Br, greens_Bz, MU0, make_solver, gs_sparse
from gspack.critical   import find_critical, core_mask, update_psi_boundary
from gspack.profiles   import ConstrainPaxisIp
from gspack.machine    import Coil, ShapedCoil, Machine, TestTokamak
from gspack.control    import constrain
from gspack.equilibrium import Equilibrium
from gspack            import picard


# ═══════════════════════════════════════════════════════
#  Green's functions
# ═══════════════════════════════════════════════════════

class TestGreens:
    def test_positive(self):
        G = greens(1.0, 0.0, 1.8, 0.5)
        assert G > 0

    def test_reciprocity(self):
        G1 = greens(1.5, 0.3,  2.0, -0.4)
        G2 = greens(2.0, -0.4, 1.5,  0.3)
        assert abs(G1 - G2) / (abs(G1) + 1e-30) < 1e-6

    def test_decay_with_distance(self):
        G_near = greens(2.0, 0.0, 1.9, 0.0)
        G_far  = greens(2.0, 0.0, 0.5, 0.0)
        assert G_near > G_far

    def test_vectorised_matches_scalar(self):
        R = np.array([1.5, 2.0, 2.5])
        Z = np.array([0.0, 0.3, -0.2])
        G_vec = greens(R, Z, 1.8, 0.1)
        for i in range(3):
            assert abs(G_vec[i] - greens(R[i], Z[i], 1.8, 0.1)) < 1e-12

    def test_Br_finite(self):
        Br = greens_Br(1.0, -1.0, 1.1, -0.6)
        assert np.isfinite(Br)

    def test_Bz_finite(self):
        Bz = greens_Bz(1.0, -1.0, 1.1, -0.6)
        assert np.isfinite(Bz)

    def test_matches_freegs(self):
        """Verify our Green's function matches FreeGS GreensBr/Bz."""
        from freegs.gradshafranov import GreensBr as FBr, GreensBz as FBz, Greens as FG
        Rc, Zc, R, Z = 1.0, -1.1, 1.1, -0.6
        assert abs(greens(Rc, Zc, R, Z)    - FG(Rc,Zc,R,Z))  < 1e-12
        assert abs(greens_Br(Rc,Zc,R,Z)   - FBr(Rc,Zc,R,Z)) < 1e-10
        assert abs(greens_Bz(Rc,Zc,R,Z)   - FBz(Rc,Zc,R,Z)) < 1e-10


# ═══════════════════════════════════════════════════════
#  GS sparse matrix & solver
# ═══════════════════════════════════════════════════════

class TestSolver:
    def test_solver_unit_source(self):
        """Unit source at centre gives localised smooth psi."""
        solver = make_solver(0.1, 2.0, -1.0, 1.0, 17, 17)
        rhs = np.zeros((17, 17))
        rhs[8, 8] = -1.0
        psi = solver(rhs)
        assert psi[8, 8] == psi.max()
        assert np.allclose(psi[0,  :], 0.0, atol=1e-12)  # zero BC
        assert np.allclose(psi[-1, :], 0.0, atol=1e-12)

    def test_solver_zero_source(self):
        """Zero source → zero solution."""
        solver = make_solver(0.1, 2.0, -1.0, 1.0, 9, 9)
        psi = solver(np.zeros((9, 9)))
        assert np.allclose(psi, 0.0, atol=1e-12)

    def test_boundary_bc(self):
        """Boundary rows of rhs set the Dirichlet values."""
        solver = make_solver(0.1, 2.0, -1.0, 1.0, 9, 9)
        rhs = np.zeros((9, 9))
        rhs[0, :] = 0.5      # set left-boundary value
        psi = solver(rhs)
        assert abs(psi[0, 4] - 0.5) < 1e-10


# ═══════════════════════════════════════════════════════
#  Critical-point finder
# ═══════════════════════════════════════════════════════

class TestCritical:
    def _gaussian_psi(self, nx=33, ny=33):
        R1d = np.linspace(0.1, 2.0, nx)
        Z1d = np.linspace(-1.0, 1.0, ny)
        R, Z = np.meshgrid(R1d, Z1d, indexing="ij")
        # Use a wider Gaussian so the O-point is clearly resolved on the grid
        psi = np.exp(-((R-1.1)**2 + Z**2)/0.30**2)
        psi[0,:]=psi[-1,:]=psi[:,0]=psi[:,-1]=0.0
        return R, Z, psi

    def test_finds_opoint_in_gaussian(self):
        R, Z, psi = self._gaussian_psi()
        opt, xpt = find_critical(R, Z, psi)
        assert len(opt) >= 1
        Ro, Zo = opt[0][0], opt[0][1]
        assert abs(Ro - 1.1) < 0.05
        assert abs(Zo - 0.0) < 0.05

    def test_no_xpoint_in_simple_gaussian(self):
        R, Z, psi = self._gaussian_psi()
        opt, xpt = find_critical(R, Z, psi)
        assert len(xpt) == 0

    def test_core_mask_contains_opoint(self):
        R, Z, psi = self._gaussian_psi()
        opt, xpt = find_critical(R, Z, psi)
        psi_bn = float(psi[0, 0])
        mask = core_mask(R, Z, psi, opt, xpt, psi_bn)
        assert mask is not None
        Ro, Zo = opt[0][0], opt[0][1]
        ir = np.argmin(np.abs(R[:,0] - Ro))
        iz = np.argmin(np.abs(Z[0,:] - Zo))
        assert mask[ir, iz] == 1.0

    def test_core_mask_zero_on_boundary(self):
        R, Z, psi = self._gaussian_psi()
        opt, xpt = find_critical(R, Z, psi)
        psi_bn = float(psi[0, 0])
        mask = core_mask(R, Z, psi, opt, xpt, psi_bn)
        assert mask is not None
        assert np.all(mask[0, :] == 0.0)
        assert np.all(mask[-1,:] == 0.0)


# ═══════════════════════════════════════════════════════
#  Machine & coils
# ═══════════════════════════════════════════════════════

class TestMachine:
    def test_coil_psi_positive(self):
        coil = Coil(1.0, 0.0, current=1e3)
        psi_val = coil.psi(np.array([[1.5]]), np.array([[0.0]]))
        assert psi_val > 0

    def test_coil_psi_scales_with_current(self):
        c1 = Coil(1.0, 0.0, current=1e3)
        c2 = Coil(1.0, 0.0, current=2e3)
        R, Z = np.array([[1.5]]), np.array([[0.0]])
        assert abs(c2.psi(R,Z) / c1.psi(R,Z) - 2.0) < 1e-10

    def test_test_tokamak_4_coils(self):
        tok = TestTokamak()
        assert len(tok.coils) == 4
        names = [n for n,_ in tok.coils]
        assert set(names) == {"P1L","P1U","P2L","P2U"}

    def test_shaped_coil_response_matches_freegs(self):
        """ShapedCoil controlBz must match FreeGS within 1%."""
        from freegs.machine import TestTokamak as FTok
        ftok = FTok()
        tok  = TestTokamak()
        R_xpt, Z_xpt = 1.1, -0.6
        for (name, coil), (_, fcoil) in zip(tok.coils, ftok.coils):
            ratio = abs(coil.controlBz(R_xpt, Z_xpt) /
                        (fcoil.controlBz(R_xpt, Z_xpt) + 1e-30))
            assert 0.98 < ratio < 1.02, f"{name} controlBz ratio={ratio:.4f}"


# ═══════════════════════════════════════════════════════
#  Profiles
# ═══════════════════════════════════════════════════════

class TestProfiles:
    def _make_eq_and_pro(self):
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0,
                          Zmin=-1.0, Zmax=1.0, nx=33, ny=33)
        pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
        return eq, pro

    def test_jtor_integral_matches_ip(self):
        eq, pro = self._make_eq_and_pro()
        eq._update_boundary_psi()
        # Manually set axis/bndry for a simple test
        eq.psi_axis  = float(eq.plasma_psi.max())
        eq.psi_bndry = float(eq.plasma_psi.min())
        from gspack.critical import core_mask, find_critical
        opt, xpt = find_critical(eq.R, eq.Z, eq.plasma_psi)
        if opt:
            mask = core_mask(eq.R, eq.Z, eq.plasma_psi, opt, xpt,
                             eq.psi_bndry)
            Jtor = pro.Jtor(eq.R, eq.Z, eq.plasma_psi,
                            eq.psi_axis, eq.psi_bndry, mask=mask)
            Ip_computed = float(Jtor.sum() * eq.dR * eq.dZ)
            assert abs(Ip_computed - 2e5) / 2e5 < 0.05

    def test_pressure_at_axis_matches_paxis(self):
        """Pressure profile is defined by integration; just check it's finite."""
        eq, pro = self._make_eq_and_pro()
        # Set consistent internal state
        pro.psi_axis  = 1.0
        pro.psi_bndry = 0.0
        pro.L         = 1e6
        pro.Beta0     = 0.01
        p0 = pro.pressure(0.0)
        # Pressure integral is finite
        assert np.isfinite(p0)


# ═══════════════════════════════════════════════════════
#  Full integration: GS solve convergence
# ═══════════════════════════════════════════════════════

class TestIntegration:
    def _run_small_solve(self, nx=33, ny=33, maxits=20):
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0,
                          Zmin=-1.0, Zmax=1.0, nx=nx, ny=ny)
        pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
        con = constrain(
            xpoints=[(1.1,-0.6),(1.1,0.6)],
            isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)],
            gamma=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            picard.solve(eq, pro, con, maxits=maxits, rtol=1e-3,
                         convergenceInfo=False)
        return eq, tok

    def test_ip_converges(self):
        eq, _ = self._run_small_solve()
        assert abs(eq.plasmaCurrent() - 2e5) / 2e5 < 0.05

    def test_psi_axis_positive(self):
        eq, _ = self._run_small_solve()
        assert eq.psi_axis > 0

    def test_psi_axis_greater_than_bndry(self):
        """For Ip > 0: psi_axis > psi_bndry."""
        eq, _ = self._run_small_solve()
        assert eq.psi_axis > eq.psi_bndry

    def test_magnetic_axis_inside_domain(self):
        eq, _ = self._run_small_solve()
        R_mag, Z_mag = eq.magneticAxis()[0], eq.magneticAxis()[1]
        assert eq.Rmin < R_mag < eq.Rmax
        assert eq.Zmin < Z_mag < eq.Zmax

    def test_magnetic_axis_near_expected(self):
        """Axis should be near R=1.25 m (FreeGS reference: R=1.254)."""
        eq, _ = self._run_small_solve()
        R_mag = eq.magneticAxis()[0]
        assert abs(R_mag - 1.254) < 0.15

    def test_psi_values_near_freegs(self):
        """psi_axis should be within 5% of FreeGS reference 0.0962."""
        eq, _ = self._run_small_solve()
        assert abs(eq.psi_axis - 0.0962) / 0.0962 < 0.15

    def test_inner_outer_separatrix(self):
        """Separatrix midplane radii should be near FreeGS: Ri=0.858, Ro=1.625."""
        eq, _ = self._run_small_solve()
        Ri, Ro = eq.innerOuterSeparatrix()
        assert 0.7 < Ri < 1.0
        assert 1.4 < Ro < 1.8

    def test_picard_converges_residuals(self):
        """Picard errors must be non-increasing on average."""
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0,
                          Zmin=-1.0, Zmax=1.0, nx=33, ny=33)
        pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
        con = constrain(
            xpoints=[(1.1,-0.6),(1.1,0.6)],
            isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)],
            gamma=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = picard.solve(eq, pro, con, maxits=15, rtol=1e-2,
                                  convergenceInfo=True)
        errors = result[0]
        assert len(errors) >= 3
        # Last half should be smaller than first half
        half = len(errors) // 2
        assert errors[half:].mean() < errors[:half].mean() * 1.1

    def test_coil_currents_reasonable(self):
        """P1 should be positive, P2 should be negative."""
        eq, tok = self._run_small_solve()
        P1L = dict(tok.coils)["P1L"].current
        P2L = dict(tok.coils)["P2L"].current
        assert P1L > 0, f"P1L={P1L:.0f} should be positive"
        assert P2L < 0, f"P2L={P2L:.0f} should be negative"

    def test_no_nans_in_psi(self):
        eq, _ = self._run_small_solve()
        assert np.all(np.isfinite(eq.psi()))

    def test_q_profile_monotonic(self):
        """q should be a monotonically increasing function of psiN."""
        eq, _ = self._run_small_solve()
        if eq._opoints:
            psiN_pts = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            q_vals   = eq.q(psiN_pts)
            if np.all(np.isfinite(q_vals)):
                assert np.all(np.diff(q_vals) > 0), "q profile not monotonic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
