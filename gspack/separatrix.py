"""
gspack.separatrix
=================
Trace the last closed flux surface (LCFS/separatrix) by poloidal angle
bisection from the O-point, matching FreeGS find_separatrix().

Key: max bisection step is limited to the distance from O-point to the
primary X-point. This keeps the tracer inside the closed plasma region
and prevents it from crossing through the X-point into the open SOL.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


def find_separatrix(eq, ntheta=360):
    """
    Return (ntheta, 2) array of (R, Z) on the LCFS at psiN = 1.

    Algorithm
    ---------
    For each poloidal angle theta:
      - Shoot a ray from (R_axis, Z_axis) in direction (sin θ, cos θ)
      - Bisect along the ray to find where psiN = 1
      - Max step length = 1.05 × dist(O-point → primary X-point)
        to stay within the closed flux region

    Parameters
    ----------
    eq     : Equilibrium object
    ntheta : number of angle samples

    Returns
    -------
    sep : ndarray (ntheta, 2)  — columns are (R, Z)
    """
    psi2d  = eq.psi()
    R1d    = eq.R[:, 0]
    Z1d    = eq.Z[0, :]
    psi_ax = eq.psi_axis
    psi_bn = eq.psi_bndry

    if psi_ax is None or psi_bn is None or not eq._opoints:
        return np.zeros((ntheta, 2))

    dpsi = psi_bn - psi_ax
    if abs(dpsi) < 1e-30:
        return np.zeros((ntheta, 2))

    # Build normalised-psi spline  (psiN = 0 at axis, 1 at boundary)
    psiN_data   = (psi2d - psi_ax) / dpsi
    psiN_spline = RectBivariateSpline(R1d, Z1d, psiN_data)

    R_ax, Z_ax = eq._opoints[0][0], eq._opoints[0][1]

    # Max step = 1.05 × distance to primary X-point.
    # For all angles, the LCFS crosses at s <= dist_xpt, so this keeps
    # the bisection inside the closed plasma region.
    if eq._xpoints:
        Rx, Zx = eq._xpoints[0][0], eq._xpoints[0][1]
        dist_xpt = np.sqrt((Rx - R_ax)**2 + (Zx - Z_ax)**2)
        max_step = dist_xpt * 1.05
    else:
        # No X-point: use domain size
        max_step = max(R1d[-1] - R_ax, R_ax - R1d[0],
                       Z1d[-1] - Z_ax, Z_ax - Z1d[0])

    # Poloidal angle grid; avoid landing exactly on X-point
    theta_grid = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]
    if eq._xpoints:
        theta_xpt = np.arctan2(Rx - R_ax, Zx - Z_ax) % (2 * np.pi)
        if np.any(np.abs(theta_grid - theta_xpt) < 1e-3):
            theta_grid = theta_grid + dtheta / 2

    sep = np.zeros((ntheta, 2))

    for k, theta in enumerate(theta_grid):
        dR_hat = np.sin(theta)
        dZ_hat = np.cos(theta)

        s_lo, s_hi = 0.0, max_step
        psiN_lo = float(psiN_spline(R_ax, Z_ax).flat[0])

        for _ in range(64):
            s_mid  = 0.5 * (s_lo + s_hi)
            R_mid  = np.clip(R_ax + s_mid * dR_hat, R1d[0] + 1e-7, R1d[-1] - 1e-7)
            Z_mid  = np.clip(Z_ax + s_mid * dZ_hat, Z1d[0] + 1e-7, Z1d[-1] - 1e-7)
            pN_mid = float(psiN_spline(R_mid, Z_mid).flat[0])

            if (psiN_lo - 1.0) * (pN_mid - 1.0) < 0:
                s_hi = s_mid
            else:
                s_lo = s_mid

        s_mid     = 0.5 * (s_lo + s_hi)
        sep[k, 0] = np.clip(R_ax + s_mid * dR_hat, R1d[0], R1d[-1])
        sep[k, 1] = np.clip(Z_ax + s_mid * dZ_hat, Z1d[0], Z1d[-1])

    return sep
