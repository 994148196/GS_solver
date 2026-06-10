"""
gspack.diagnostics — Poincaré sections, flux coordinates, MHD indicators
"""
import numpy as np


def poincare_section(eq, R0_list=None, Z0_list=None, start_points=None, n_turns=200, n_steps_per_turn=200):
    """
    Poincaré section: trace field lines and record each toroidal crossing.

    Field line equations (φ as parameter):
        dR/dφ = R·Br/Bφ
        dZ/dφ = R·Bz/Bφ

    Parameters
    ----------
    eq           : Equilibrium
    R0_list, Z0_list : starting points (same length)
    n_turns      : toroidal turns to trace
    n_steps_per_turn : integration steps per turn

    Returns
    -------
    List of (R_pts, Z_pts) arrays, one per starting point.
    """
    from scipy.integrate import solve_ivp
    from .backend import to_numpy

    results = []

    for R0, Z0 in zip(R0_list, Z0_list):

        def rhs(phi, y):
            R, Z = float(y[0]), float(y[1])
            R = max(R, 0.01)
            Br  = float(to_numpy(eq.Br(R, Z)).flat[0])
            Bz  = float(to_numpy(eq.Bz(R, Z)).flat[0])
            Bt  = float(to_numpy(eq.Btor(np.array([[R]]), np.array([[Z]]))).flat[0])
            denom = Bt + 1e-30
            return [R*Br/denom, R*Bz/denom]

        phi_end  = 2*np.pi * n_turns
        t_eval   = np.linspace(0, phi_end, n_turns * n_steps_per_turn)

        try:
            sol = solve_ivp(rhs, [0, phi_end], [R0, Z0],
                            t_eval=t_eval, method='RK45',
                            rtol=1e-6, atol=1e-8,
                            dense_output=False, max_step=2*np.pi/n_steps_per_turn)
            # Sample at each 2π crossing
            idx    = np.arange(0, len(t_eval), n_steps_per_turn)
            R_pts  = sol.y[0, idx]
            Z_pts  = sol.y[1, idx]
        except Exception:
            R_pts, Z_pts = np.array([R0]), np.array([Z0])

        results.append((R_pts, Z_pts))

    return results


def flux_surface_coordinates(eq, n_psi=50, n_theta=128):
    """
    Build flux-surface coordinate grid (R, Z) as functions of (ψN, θ).

    Returns a dict with:
        psiN   (n_psi,)
        theta  (n_theta,)
        R      (n_psi, n_theta)
        Z      (n_psi, n_theta)
        B      (n_psi, n_theta)  total field
        Bpol   (n_psi, n_theta)  poloidal field
        q      (n_psi,)
        dV_dpsiN (n_psi,)  flux-surface volume element

    Suitable as input for transport codes (GENE, GS2, BOUT++).
    """
    from .separatrix import find_separatrix
    from .backend import to_numpy

    psiN_arr  = np.linspace(0.02, 0.98, n_psi)
    theta_arr = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

    R_fs   = np.zeros((n_psi, n_theta))
    Z_fs   = np.zeros((n_psi, n_theta))
    Bpol_f = np.zeros((n_psi, n_theta))
    Bt_f   = np.zeros((n_psi, n_theta))

    for i, psiN in enumerate(psiN_arr):
        # Temporarily adjust psi_bndry to trace this surface
        eq_orig_bndry = eq.psi_bndry
        psi_target = eq.psi_axis + psiN * (eq.psi_bndry - eq.psi_axis)

        # Use separatrix tracer at psiN (not 1.0)
        from .separatrix import find_separatrix as _sep
        from scipy.interpolate import RectBivariateSpline
        import copy

        # Temporarily trick psi_bndry so find_separatrix traces psiN surface
        eq_temp = copy.copy(eq)
        eq_temp.psi_bndry = psi_target
        surf = _sep(eq_temp, ntheta=n_theta)

        R_fs[i, :] = surf[:, 0]
        Z_fs[i, :] = surf[:, 1]

        Br_s = to_numpy(eq.Br(R_fs[i,:], Z_fs[i,:]))
        Bz_s = to_numpy(eq.Bz(R_fs[i,:], Z_fs[i,:]))
        Bpol_f[i, :] = np.sqrt(Br_s**2 + Bz_s**2)
        Bt_f[i,:]    = to_numpy(eq.Btor(R_fs[i,:].reshape(1,-1),
                                         Z_fs[i,:].reshape(1,-1))).ravel()

    B_fs = np.sqrt(Bpol_f**2 + Bt_f**2)

    # Volume element dV/dψN ≈ 2π ∮ R/Bpol dl
    dV_dpsi = np.zeros(n_psi)
    for i in range(n_psi):
        dR  = np.roll(R_fs[i], -1) - R_fs[i]
        dZZ = np.roll(Z_fs[i], -1) - Z_fs[i]
        dl  = np.sqrt(dR**2 + dZZ**2)
        dV_dpsi[i] = 2*np.pi * np.sum(R_fs[i] * dl / (Bpol_f[i] + 1e-30))

    return {
        'psiN': psiN_arr, 'theta': theta_arr,
        'R': R_fs, 'Z': Z_fs,
        'B': B_fs, 'Bpol': Bpol_f,
        'q': eq.q(psiN_arr),
        'dV_dpsiN': dV_dpsi,
    }


def mhd_stability_indicators(eq):
    """
    Compute common MHD stability indicators.

    Returns a dict:
        q=1 surface psiN     — sawtooth / internal kink
        q=2 surface psiN     — NTM (2,1 mode)
        q=3 surface psiN     — NTM (3,2 mode)
        q_axis               — safety factor on axis
        q_95                 — safety factor at ψN=0.95
        Greenwald fraction   — n/n_GW
        betaN                — Troyon normalised beta
        Troyon margin        — 2.8 - betaN (positive = stable)
        li                   — normalised internal inductance
    """
    psiN_dense = np.linspace(0.02, 0.98, 300)
    q_dense    = eq.q(psiN_dense)

    def find_q_surface(q_target):
        """Return psiN where q = q_target, or NaN if not found."""
        idx = np.where(np.diff(np.sign(q_dense - q_target)))[0]
        if len(idx) == 0:
            return float('nan')
        i = idx[0]
        frac = (q_target - q_dense[i]) / (q_dense[i+1] - q_dense[i] + 1e-30)
        return float(psiN_dense[i] + frac*(psiN_dense[i+1]-psiN_dense[i]))

    q_axis = float(eq.q(np.array([0.01]))[0])
    q_95   = float(eq.q(np.array([0.95]))[0])
    a      = eq.minorRadius()
    Ip     = eq.plasmaCurrent()
    n_GW   = Ip / (np.pi * (a + 1e-30)**2 * 1e20)   # 10²⁰ m⁻³

    return {
        'q_axis':           q_axis,
        'q_95':             q_95,
        'q=1 surface psiN': find_q_surface(1.0),
        'q=2 surface psiN': find_q_surface(2.0),
        'q=3 surface psiN': find_q_surface(3.0),
        'Greenwald_density_limit_1e20': float(n_GW),
        'betaN':            float(eq.betaN()),
        'Troyon_margin':    float(2.8 - eq.betaN()),
        'li':               float(eq.internalInductance()),
        'kappa':            float(eq.elongation()),
        'delta':            float(eq.triangularity()),
        'minor_radius_m':   float(a),
    }

# ── Clean aliases ────────────────────────────────────────────────────────────
mhd_stability  = mhd_stability_indicators
flux_coordinates = flux_surface_coordinates
