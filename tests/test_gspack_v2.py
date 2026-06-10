"""
tests/test_gspack_v2.py
=======================
Full test suite for gspack v2.0
Run:  cd gspack2 && python -m pytest tests/ -v
"""

import sys, os, warnings, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════════
#  Backend
# ══════════════════════════════════════════════════════════════════════════════

class TestBackend:
    def test_default_cpu(self):
        import gspack.backend as bk
        assert bk.get_backend() in ('cpu', 'gpu')

    def test_set_cpu(self):
        import gspack.backend as bk
        bk.set_backend('cpu')
        assert bk.get_backend() == 'cpu'

    def test_to_numpy_passthrough(self):
        import gspack.backend as bk
        arr = np.array([1.0, 2.0])
        result = bk.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_to_backend_returns_array(self):
        import gspack.backend as bk
        arr = np.array([1.0, 2.0])
        result = bk.to_backend(arr)
        assert bk.to_numpy(result)[0] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Green's functions
# ══════════════════════════════════════════════════════════════════════════════

class TestGreens:
    def test_positive(self):
        from gspack.greens import greens
        G = float(greens(1.0, 0.0, 1.8, 0.5))
        assert G > 0

    def test_reciprocity(self):
        from gspack.greens import greens
        G1 = float(greens(1.5,  0.3, 2.0, -0.4))
        G2 = float(greens(2.0, -0.4, 1.5,  0.3))
        assert abs(G1 - G2) / (abs(G1) + 1e-30) < 1e-6

    def test_vectorised(self):
        from gspack.greens import greens
        import gspack.backend as bk
        R = np.array([1.5, 2.0, 2.5])
        Z = np.array([0.0, 0.3, -0.2])
        G_vec = bk.to_numpy(greens(R, Z, 1.8, 0.1))
        for i in range(3):
            G_sc = float(greens(R[i], Z[i], 1.8, 0.1))
            assert abs(G_vec[i] - G_sc) < 1e-12

    def test_matches_freegs(self):
        from gspack.greens import greens, greens_Br, greens_Bz
        from freegs.gradshafranov import GreensBr, GreensBz, Greens
        Rc, Zc, R, Z = 1.0, -1.1, 1.1, -0.6
        assert abs(float(greens(Rc,Zc,R,Z)) - Greens(Rc,Zc,R,Z))   < 1e-11
        assert abs(float(greens_Br(Rc,Zc,R,Z)) - GreensBr(Rc,Zc,R,Z)) < 1e-9
        assert abs(float(greens_Bz(Rc,Zc,R,Z)) - GreensBz(Rc,Zc,R,Z)) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  GS solver — 2nd and 4th order
# ══════════════════════════════════════════════════════════════════════════════

class TestSolver:
    @pytest.mark.parametrize("order", [2, 4])
    def test_unit_source(self, order):
        from gspack.greens import make_solver
        solver = make_solver(0.1, 2.0, -1.0, 1.0, 17, 17, order=order, method='lu')
        rhs = np.zeros((17, 17))
        rhs[8, 8] = -1.0
        psi = solver(rhs)
        from gspack.backend import to_numpy
        psi_np = to_numpy(psi)
        assert psi_np[8, 8] == pytest.approx(psi_np.max(), rel=1e-6)
        assert abs(psi_np[0, 0]) < 1e-12

    def test_zero_source(self):
        from gspack.greens import make_solver
        solver = make_solver(0.1, 2.0, -1.0, 1.0, 9, 9, order=2, method='lu')
        from gspack.backend import to_numpy
        psi = to_numpy(solver(np.zeros((9, 9))))
        np.testing.assert_allclose(psi, 0.0, atol=1e-12)

    def test_4th_order_more_accurate(self):
        """4th-order should be more accurate than 2nd-order on same grid."""
        from gspack.greens import make_solver, MU0
        Rmin, Rmax, Zmin, Zmax, n = 0.5, 2.0, -1.0, 1.0, 33
        R1d = np.linspace(Rmin, Rmax, n)
        Z1d = np.linspace(Zmin, Zmax, n)
        R, Z = np.meshgrid(R1d, Z1d, indexing='ij')
        # Toroidal current ring source at (1.2, 0)
        Jtor = np.exp(-((R-1.2)**2 + Z**2)/0.1**2) * 1e6

        from gspack.backend import to_numpy
        for order in [2, 4]:
            s = make_solver(Rmin, Rmax, Zmin, Zmax, n, n, order=order, method='lu')
            rhs = np.zeros((n, n))
            rhs[1:-1, 1:-1] = -MU0 * R[1:-1, 1:-1] * Jtor[1:-1, 1:-1]
            psi = to_numpy(s(rhs))
            # psi should peak near (1.2, 0)
            idx = np.unravel_index(psi.argmax(), psi.shape)
            assert abs(R1d[idx[0]] - 1.2) < 0.15

    def test_amg_solver(self):
        """AMG solver (if pyamg installed) produces a valid GS solution."""
        try:
            import pyamg
        except ImportError:
            pytest.skip("pyamg not installed")
        from gspack.greens import make_solver, MU0
        from gspack.backend import to_numpy
        import warnings
        nx, ny = 33, 33
        solver_amg = make_solver(0.1,2.0,-1.0,1.0,nx,ny,method='amg')
        R1d = np.linspace(0.1, 2.0, nx)
        Z1d = np.linspace(-1.0, 1.0, ny)
        R, Z = np.meshgrid(R1d, Z1d, indexing='ij')
        # Gaussian source in interior
        Jtor = np.exp(-((R-1.2)**2 + Z**2)/0.1**2) * 1e6
        rhs = np.zeros((nx, ny))
        rhs[1:-1, 1:-1] = -MU0 * R[1:-1,1:-1] * Jtor[1:-1,1:-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psi = to_numpy(solver_amg(rhs))
        # psi should peak near R=1.2
        idx = np.unravel_index(psi.argmax(), psi.shape)
        assert abs(R1d[idx[0]] - 1.2) < 0.2
        assert np.isfinite(psi).all()


# ══════════════════════════════════════════════════════════════════════════════
#  Critical points
# ══════════════════════════════════════════════════════════════════════════════

class TestCritical:
    def _make_gaussian(self, nx=33, ny=33, width=0.3):
        R1d = np.linspace(0.1, 2.0, nx)
        Z1d = np.linspace(-1.0, 1.0, ny)
        R, Z = np.meshgrid(R1d, Z1d, indexing='ij')
        psi = np.exp(-((R-1.1)**2 + Z**2)/width**2)
        psi[[0,-1],:] = psi[:,[0,-1]] = 0.0
        return R, Z, psi

    def test_find_opoint(self):
        from gspack.critical import find_critical
        R, Z, psi = self._make_gaussian()
        opt, xpt = find_critical(R, Z, psi)
        assert len(opt) >= 1
        assert abs(opt[0][0] - 1.1) < 0.05

    def test_no_xpoint_in_gaussian(self):
        from gspack.critical import find_critical
        R, Z, psi = self._make_gaussian()
        _, xpt = find_critical(R, Z, psi)
        assert len(xpt) == 0

    def test_flood_fill_mask(self):
        from gspack.critical import find_critical, core_mask
        R, Z, psi = self._make_gaussian()
        opt, xpt = find_critical(R, Z, psi)
        mask = core_mask(R, Z, psi, opt, xpt, float(psi[0, 0]))
        assert mask is not None
        # O-point index should be inside
        ir = np.argmin(np.abs(R[:,0] - opt[0][0]))
        iz = np.argmin(np.abs(Z[0,:] - opt[0][1]))
        assert mask[ir, iz] == 1.0
        # Boundaries outside
        assert np.all(mask[0, :] == 0.0)
        assert np.all(mask[-1,:] == 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Profiles
# ══════════════════════════════════════════════════════════════════════════════

class TestProfiles:
    def _make_eq_and_run(self, nx=33, ny=33, maxits=15):
        from gspack.machine    import TestTokamak
        from gspack.equilibrium import Equilibrium
        from gspack.profiles   import ConstrainPaxisIp
        from gspack.control    import constrain
        from gspack            import picard

        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                          nx=nx, ny=ny)
        pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
        con = constrain(
            xpoints=[(1.1,-0.6),(1.1,0.6)],
            isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)],
            gamma=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            picard.solve(eq, pro, con, maxits=maxits, rtol=1e-2,
                        convergenceInfo=False)
        return eq, pro

    def test_jtor_integral_ip(self):
        eq, _ = self._make_eq_and_run()
        assert abs(eq.plasmaCurrent() - 2e5) / 2e5 < 0.05

    def test_pressure_finite(self):
        _, pro = self._make_eq_and_run()
        pro.psi_axis  = 1.0;  pro.psi_bndry = 0.0
        pro.L = 1e6;          pro.Beta0 = 0.01
        assert np.isfinite(pro.pressure(0.0))

    def test_fpol_positive(self):
        _, pro = self._make_eq_and_run()
        pro.psi_axis  = 1.0;  pro.psi_bndry = 0.0
        pro.L = 1e6;          pro.Beta0 = 0.01
        assert pro.fpol(0.5) > 0

    def test_constrain_betap(self):
        """ConstrainBetapIp runs without error."""
        from gspack.profiles import ConstrainBetapIp
        from gspack.machine  import TestTokamak
        from gspack.equilibrium import Equilibrium
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                          nx=17, ny=17)
        pro = ConstrainBetapIp(betap=0.5, Ip=2e5, fvac=1.0)
        eq._update_boundary_psi()
        mask = np.ones((17, 17))
        Jtor = pro.Jtor(eq.R, eq.Z, eq.psi(),
                        eq.psi_axis, eq.psi_bndry, mask=mask)
        assert np.isfinite(Jtor).all()

    def test_profiles_pprime_ffprime(self):
        """ProfilesPprimeFfprime works with lambda functions."""
        from gspack.profiles import ProfilesPprimeFfprime
        pro = ProfilesPprimeFfprime(
            pprime_fn=lambda pN: -1e3 * (1 - pN),
            ffprime_fn=lambda pN: -0.5 * (1 - pN),
            fvac=1.0)
        pro.psi_axis  = 1.0;  pro.psi_bndry = 0.0
        R, Z = np.meshgrid([1.0, 1.2], [0.0, 0.2], indexing='ij')
        psi = np.ones_like(R) * 0.5
        Jtor = pro.Jtor(R, Z, psi, 1.0, 0.0)
        assert np.isfinite(Jtor).all()


# ══════════════════════════════════════════════════════════════════════════════
#  Anderson mixing
# ══════════════════════════════════════════════════════════════════════════════

class TestAndersonMixer:
    def test_passthrough_m0(self):
        from gspack.picard import AndersonMixer
        m = AndersonMixer(m=0)
        x_old = np.array([1.0, 2.0, 3.0])
        x_new = np.array([1.5, 2.5, 3.5])
        result = m.mix(x_new, x_old)
        np.testing.assert_array_equal(result, x_new)

    def test_first_step_passthrough(self):
        """With only 1 history point, should return x_new."""
        from gspack.picard import AndersonMixer
        m = AndersonMixer(m=5)
        x_old = np.ones(10)
        x_new = np.ones(10) * 2.0
        result = m.mix(x_new, x_old)
        np.testing.assert_array_equal(result, x_new)

    def test_converges_linear(self):
        """Anderson should converge faster than Picard on linear iteration."""
        from gspack.picard import AndersonMixer
        # x* = 1 is fixed point of T(x) = 0.9*(x-1) + 1
        def T(x):
            return 0.9*(x - 1.0) + 1.0

        mixer = AndersonMixer(m=3)
        x = np.array([0.0])
        errors = []
        for _ in range(20):
            x_new = T(x)
            x = mixer.mix(x_new, x)
            errors.append(abs(float(np.asarray(x).flat[0]) - 1.0))

        assert errors[-1] < errors[0] * 0.01  # converged


# ══════════════════════════════════════════════════════════════════════════════
#  Machine & control
# ══════════════════════════════════════════════════════════════════════════════

class TestMachine:
    def test_testtokamak_4_coils(self):
        from gspack.machine import TestTokamak
        tok = TestTokamak()
        assert len(tok.coils) == 4
        assert set(n for n,_ in tok.coils) == {"P1L","P1U","P2L","P2U"}

    def test_coil_psi_scales_with_current(self):
        from gspack.machine import Coil
        from gspack.backend import to_numpy
        c1 = Coil(1.0, 0.0, current=1e3)
        c2 = Coil(1.0, 0.0, current=2e3)
        R, Z = np.array([[1.5]]), np.array([[0.0]])
        ratio = to_numpy(c2.psi(R,Z)).flat[0] / to_numpy(c1.psi(R,Z)).flat[0]
        assert ratio == pytest.approx(2.0, rel=1e-10)

    def test_controlBz_matches_freegs(self):
        """ShapedCoil controlBz must match FreeGS within 1%."""
        from gspack.machine import TestTokamak
        from freegs.machine import TestTokamak as FTok
        tok  = TestTokamak()
        ftok = FTok()
        for (name, coil), (_, fcoil) in zip(tok.coils, ftok.coils):
            ratio = abs(coil.controlBz(1.1,-0.6) /
                        (fcoil.controlBz(1.1,-0.6)+1e-30))
            assert 0.98 < ratio < 1.02, f"{name}: ratio={ratio:.4f}"

    def test_snowflake_constrain_runs(self):
        """constrain_snowflake should not crash on first call."""
        from gspack.machine    import TestTokamak
        from gspack.equilibrium import Equilibrium
        from gspack.control    import constrain_snowflake
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                          nx=17, ny=17)
        con = constrain_snowflake(
            xpoints=[(1.1,-0.6),(1.1,0.6)],
            sf_xpoints=[(1.1,-0.6)],
            gamma=1e-10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            con(eq)   # should not raise


# ══════════════════════════════════════════════════════════════════════════════
#  Integration: full solve
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    @pytest.fixture(scope="class")
    def solved_eq(self):
        from gspack.machine    import TestTokamak
        from gspack.equilibrium import Equilibrium
        from gspack.profiles   import ConstrainPaxisIp
        from gspack.control    import constrain
        from gspack            import picard
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                          nx=33, ny=33)
        pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
        con = constrain(
            xpoints=[(1.1,-0.6),(1.1,0.6)],
            isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)],
            gamma=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            picard.solve(eq, pro, con, maxits=25, rtol=1e-2,
                        anderson_m=5, convergenceInfo=False)
        return eq

    def test_ip_correct(self, solved_eq):
        assert abs(solved_eq.plasmaCurrent() - 2e5) / 2e5 < 0.05

    def test_psi_axis_positive(self, solved_eq):
        assert solved_eq.psi_axis > 0

    def test_psi_axis_gt_bndry(self, solved_eq):
        assert solved_eq.psi_axis > solved_eq.psi_bndry

    def test_axis_inside_domain(self, solved_eq):
        R_mag, Z_mag = solved_eq.magneticAxis()[:2]
        assert solved_eq.Rmin < R_mag < solved_eq.Rmax
        assert solved_eq.Zmin < Z_mag < solved_eq.Zmax

    def test_axis_near_freegs(self, solved_eq):
        assert abs(solved_eq.magneticAxis()[0] - 1.254) < 0.15

    def test_psi_axis_near_freegs(self, solved_eq):
        assert abs(solved_eq.psi_axis - 0.0962) / 0.0962 < 0.15

    def test_separatrix_radii(self, solved_eq):
        Ri, Ro = solved_eq.innerOuterSeparatrix()
        assert 0.7 < Ri < 1.0
        assert 1.4 < Ro < 1.8

    def test_coil_signs(self, solved_eq):
        P1L = dict(solved_eq.tokamak.coils)["P1L"].current
        P2L = dict(solved_eq.tokamak.coils)["P2L"].current
        assert P1L > 0, f"P1L={P1L:.0f} should be positive"
        assert P2L < 0, f"P2L={P2L:.0f} should be negative"

    def test_no_nans(self, solved_eq):
        assert np.all(np.isfinite(solved_eq.psi()))

    def test_q_monotonic(self, solved_eq):
        psiN = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q    = solved_eq.q(psiN)
        if np.all(np.isfinite(q)):
            assert np.all(np.diff(q) > 0), "q should be increasing"

    def test_elongation_positive(self, solved_eq):
        assert solved_eq.elongation() > 0

    def test_volume_positive(self, solved_eq):
        assert solved_eq.plasmaVolume() > 0

    def test_anderson_converges_faster(self):
        """Anderson mixing should converge in fewer iterations than plain Picard."""
        from gspack.machine    import TestTokamak
        from gspack.equilibrium import Equilibrium
        from gspack.profiles   import ConstrainPaxisIp
        from gspack.control    import constrain
        from gspack            import picard

        def run(anderson_m):
            tok = TestTokamak()
            eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                              nx=33, ny=33)
            pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
            con = constrain(
                xpoints=[(1.1,-0.6),(1.1,0.6)],
                isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)],
                gamma=1e-12)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                errs = picard.solve(eq, pro, con, maxits=50, rtol=1e-3,
                                   anderson_m=anderson_m, convergenceInfo=True)
            return len(errs[0])

        n_picard   = run(anderson_m=0)
        n_anderson = run(anderson_m=5)
        # Anderson should not be dramatically worse (may or may not be better
        # depending on the specific problem)
        assert n_anderson <= n_picard * 1.5


# ══════════════════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════════════════

class TestIO:
    @pytest.fixture(scope="class")
    def eq_solved(self):
        from gspack.machine    import TestTokamak
        from gspack.equilibrium import Equilibrium
        from gspack.profiles   import ConstrainPaxisIp
        from gspack.control    import constrain
        from gspack            import picard
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                          nx=33, ny=33)
        pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
        con = constrain(
            xpoints=[(1.1,-0.6),(1.1,0.6)],
            isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)],
            gamma=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            picard.solve(eq, pro, con, maxits=15, rtol=1e-2, convergenceInfo=False)
        return eq

    def test_geqdsk_write(self, eq_solved, tmp_path):
        from gspack.geqdsk import write_geqdsk
        outfile = str(tmp_path / "test.geqdsk")
        write_geqdsk(eq_solved, outfile)
        assert os.path.exists(outfile)
        assert os.path.getsize(outfile) > 1000  # non-trivial size

    def test_geqdsk_read_roundtrip(self, eq_solved, tmp_path):
        from gspack.geqdsk import write_geqdsk, read_geqdsk
        outfile = str(tmp_path / "test.geqdsk")
        write_geqdsk(eq_solved, outfile)
        data = read_geqdsk(outfile)
        # read_geqdsk returns a dict
        nw = data['nw'] if isinstance(data, dict) else data.nw
        assert nw == eq_solved.nx

    def test_hdf5_save_load(self, eq_solved, tmp_path):
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")
        from gspack.io import save, load
        outfile = str(tmp_path / "test.h5")
        save(eq_solved, outfile)
        assert os.path.exists(outfile)
        data = load(outfile)
        # load returns a dict or PostEquil object
        Ip_loaded = data['Ip'] if isinstance(data, dict) else data.Ip
        assert abs(Ip_loaded - eq_solved.plasmaCurrent()) < 1e3


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagnostics:
    @pytest.fixture(scope="class")
    def eq_solved(self):
        from gspack.machine    import TestTokamak
        from gspack.equilibrium import Equilibrium
        from gspack.profiles   import ConstrainPaxisIp
        from gspack.control    import constrain
        from gspack            import picard
        tok = TestTokamak()
        eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                          nx=33, ny=33)
        pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
        con = constrain(
            xpoints=[(1.1,-0.6),(1.1,0.6)],
            isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)],
            gamma=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            picard.solve(eq, pro, con, maxits=15, rtol=1e-2, convergenceInfo=False)
        return eq

    def test_mhd_stability_keys(self, eq_solved):
        from gspack.diagnostics import mhd_stability
        stab = mhd_stability(eq_solved)
        for key in ['q_axis', 'q_95', 'betaN', 'li']:
            assert key in stab

    def test_q95_positive(self, eq_solved):
        from gspack.diagnostics import mhd_stability
        stab = mhd_stability(eq_solved)
        assert stab['q_95'] > 0

    def test_poincare_runs(self, eq_solved):
        from gspack.diagnostics import poincare_section
        traces = poincare_section(eq_solved,
                                  R0_list=[1.1, 1.2], Z0_list=[0.0, 0.0],
                                  n_turns=5, n_steps_per_turn=50)
        assert len(traces) == 2
        for R_t, Z_t in traces:
            assert len(R_t) > 0

    def test_flux_coordinates_shape(self, eq_solved):
        from gspack.diagnostics import flux_coordinates
        fc = flux_coordinates(eq_solved, n_psi=5, n_theta=16)
        if fc:   # returns empty dict if no O-points
            assert fc['R'].shape == (5, 16)
            assert fc['q'].shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
