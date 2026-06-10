"""
gspack.boundary
===============
Von Hagenow free-boundary condition — GPU transparent.

The GS solve runs on CPU (scipy). rhs and results are transferred
to/from the active backend (GPU/CPU) transparently.

All intermediate numpy arrays stay on CPU since scipy integration
(romb/trapz) requires numpy.
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

    wL = (dUdn_L / R1d[0])  * dZ
    wR = (dUdn_R / R1d[-1]) * dZ
    wB = (dUdn_B / R1d)     * dR
    wT = (dUdn_T / R1d)     * dR

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
