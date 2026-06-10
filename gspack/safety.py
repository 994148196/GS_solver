"""
gspack.safety
=============
Safety factor q(ψN) via flux-surface tracing.

For each flux surface ψN:
  1. Trace (R,Z) of the closed flux surface by bisecting along
     radial rays from the O-point at each poloidal angle θ.
  2. q = (1/2π) ∮ f dl / (R² Bpol)
       = (f / 2π) Σ |dl| / (R Bpol)
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def _find_flux_surface(psi_spline, R_axis, Z_axis, theta_arr, psi_target, R1d, Z1d):
    """
    Trace the flux surface ψ = psi_target by bisection from the O-point.
    Returns (R_surf, Z_surf) arrays of length ntheta.
    """
    ntheta = len(theta_arr)
    R_surf = np.zeros(ntheta)
    Z_surf = np.zeros(ntheta)

    R_extent = max(R1d[-1] - R_axis, R_axis - R1d[0])
    Z_extent = max(Z1d[-1] - Z_axis, Z_axis - Z1d[0])
    max_step = max(R_extent, Z_extent)

    psi_lo = float(psi_spline(R_axis, Z_axis).flat[0])

    for k, theta in enumerate(theta_arr):
        dR_hat = np.sin(theta)
        dZ_hat = np.cos(theta)

        s_lo, s_hi = 0.0, max_step

        for _ in range(60):
            s_mid = 0.5 * (s_lo + s_hi)
            R_mid = np.clip(R_axis + s_mid * dR_hat, R1d[0] + 1e-6, R1d[-1] - 1e-6)
            Z_mid = np.clip(Z_axis + s_mid * dZ_hat, Z1d[0] + 1e-6, Z1d[-1] - 1e-6)
            psi_mid = float(psi_spline(R_mid, Z_mid).flat[0])

            if (psi_lo - psi_target) * (psi_mid - psi_target) < 0:
                s_hi = s_mid
            else:
                s_lo = s_mid

        s_mid = 0.5 * (s_lo + s_hi)
        R_surf[k] = np.clip(R_axis + s_mid * dR_hat, R1d[0], R1d[-1])
        Z_surf[k] = np.clip(Z_axis + s_mid * dZ_hat, Z1d[0], Z1d[-1])

    return R_surf, Z_surf


def find_safety(eq, psiN_arr, ntheta=128):
    """
    Compute q at each psiN in psiN_arr.

    Parameters
    ----------
    eq       : Equilibrium
    psiN_arr : 1-D array of ψN values (0=axis, 1=boundary)
    ntheta   : poloidal angle resolution

    Returns
    -------
    q_arr : 1-D array, same length as psiN_arr
    """
    psi2d  = eq.psi()
    R1d    = eq.R[:, 0]
    Z1d    = eq.Z[0, :]
    psi_ax = eq.psi_axis
    psi_bn = eq.psi_bndry

    if psi_ax is None or psi_bn is None:
        return np.full(len(psiN_arr), np.nan)

    psi_spline = RectBivariateSpline(R1d, Z1d, psi2d)

    if not eq._opoints:
        return np.full(len(psiN_arr), np.nan)
    R_axis, Z_axis = eq._opoints[0][0], eq._opoints[0][1]

    # Poloidal angle grid
    theta_arr = np.linspace(0, 2 * np.pi, ntheta, endpoint=False) + 0.01

    q_arr = np.zeros(len(psiN_arr))

    for idx, psiN_val in enumerate(psiN_arr):
        psiN_val   = float(np.clip(psiN_val, 0.01, 0.99))
        psi_target = psi_ax + psiN_val * (psi_bn - psi_ax)

        R_surf, Z_surf = _find_flux_surface(
            psi_spline, R_axis, Z_axis, theta_arr, psi_target, R1d, Z1d)

        # Arc length elements  dl
        dR_arc = np.roll(R_surf, -1) - R_surf
        dZ_arc = np.roll(Z_surf, -1) - Z_surf
        dl     = np.sqrt(dR_arc**2 + dZ_arc**2)

        # Poloidal field along surface
        Br_s   = eq.Br(R_surf, Z_surf)
        Bz_s   = eq.Bz(R_surf, Z_surf)
        Bpol_s = np.sqrt(Br_s**2 + Bz_s**2) + 1e-30

        # f = R*Bφ = fpol(ψN)
        f_val = eq._profiles.fpol(psiN_val) if eq._profiles else 1.0

        # q = (f / 2π) ∮ dl / (R Bpol)
        q_arr[idx] = f_val / (2.0 * np.pi) * np.sum(dl / (R_surf * Bpol_s))

    return q_arr
