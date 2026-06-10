"""
gspack.greens
=============
Green's function + GS operator matrix.

GPU transparency: all array ops use get_xp() called inside each function.
The GS matrix and LU/AMG solvers always run on CPU (scipy requirement).
rhs is converted numpy→GPU after solving.
"""

import numpy as np
from scipy.special import ellipk as _cpu_ellipk, ellipe as _cpu_ellipe
from scipy.sparse import lil_matrix, eye as speye

from .backend import (MU0, get_xp, to_numpy, to_backend, asarray,
                      ellipk_compat, ellipe_compat)


# ─────────────────────────────────────────────────────────────────────────────
#  Green's function  (GPU-transparent)
# ─────────────────────────────────────────────────────────────────────────────

def greens(Rc, Zc, R, Z):
    """
    Poloidal flux at (R,Z) from unit toroidal current at (Rc,Zc).
    Fully broadcastable.  Runs on GPU when backend='gpu'.
    
    All arguments are converted to the active backend device before
    computation, so mixing numpy scalars with cupy arrays is safe.
    """
    xp = get_xp()
    # Convert everything to backend arrays — prevents CuPy "Unsupported type"
    # when Rc/Zc are Python floats or numpy scalars while R/Z are CuPy arrays
    R  = asarray(R)
    Rc = asarray(Rc)
    Z  = asarray(Z)
    Zc = asarray(Zc)

    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = xp.clip(k2, 1e-10, 1.0 - 1e-10)
    k  = xp.sqrt(k2)

    Kk = ellipk_compat(k2)
    Ek = ellipe_compat(k2)

    return (MU0 / (2.0 * xp.pi)) * xp.sqrt(R * Rc) \
           * ((2.0 - k2) * Kk - 2.0 * Ek) / k


def greens_Bz(Rc, Zc, R, Z, eps=1e-3):
    """Bz = (1/R) ∂G/∂R"""
    xp = get_xp()
    R  = asarray(R)
    return (greens(Rc, Zc, R + eps, Z) - greens(Rc, Zc, R - eps, Z)) \
           / (2.0 * eps * R)


def greens_Br(Rc, Zc, R, Z, eps=1e-3):
    """Br = -(1/R) ∂G/∂Z"""
    xp = get_xp()
    Z  = asarray(Z)
    R  = asarray(R)
    return (greens(Rc, Zc, R, Z - eps) - greens(Rc, Zc, R, Z + eps)) \
           / (2.0 * eps * R)


# ─────────────────────────────────────────────────────────────────────────────
#  Sparse GS operator matrices  (always CPU / scipy)
# ─────────────────────────────────────────────────────────────────────────────

def gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, nx, ny):
    """2nd-order centred FDM sparse matrix for Δ*."""
    dR = (Rmax - Rmin) / (nx - 1)
    dZ = (Zmax - Zmin) / (ny - 1)
    N  = nx * ny
    A  = speye(N, format="lil")
    iR2, iZ2 = 1.0 / dR**2, 1.0 / dZ**2

    for i in range(1, nx - 1):
        R = Rmin + dR * i
        for j in range(1, ny - 1):
            row = i * ny + j
            A[row, row - 1]  = iZ2
            A[row, row - ny] = iR2 + 1.0 / (2.0 * R * dR)
            A[row, row]      = -2.0 * (iR2 + iZ2)
            A[row, row + ny] = iR2 - 1.0 / (2.0 * R * dR)
            A[row, row + 1]  = iZ2
    return A.tocsr()


def gs_sparse_4th(Rmin, Rmax, Zmin, Zmax, nx, ny):
    """4th-order FDM sparse matrix for Δ*. Truncation error O(h⁴)."""
    dR = (Rmax - Rmin) / (nx - 1)
    dZ = (Zmax - Zmin) / (ny - 1)
    N  = nx * ny
    A  = lil_matrix((N, N))
    iR2, iZ2 = 1.0 / dR**2, 1.0 / dZ**2

    c2 = [-1/12, 4/3, -5/2, 4/3, -1/12]   # d²/dx² centred 4th-order
    c1 = [ 1/12,-2/3,   0,  2/3, -1/12]   # d/dx  centred 4th-order
    os2 = [10/12,-15/12,-4/12,14/12,-6/12,1/12]  # one-sided d²/dx²
    os1 = [-3/12,-10/12,18/12,-6/12,1/12]         # one-sided d/dx

    for i in range(1, nx - 1):
        R = Rmin + dR * i
        for j in range(1, ny - 1):
            row = i * ny + j
            # d²/dZ²
            if j == 1:
                for k, w in enumerate(os2):
                    A[row, row + (k-1)] += w * iZ2
            elif j == ny - 2:
                for k, w in enumerate(os2):
                    A[row, row - (k-1)] += w * iZ2
            else:
                for k, w in enumerate(c2):
                    A[row, row + (k-2)] += w * iZ2
            # d²/dR² - (1/R) d/dR
            if i == 1:
                for k, w in enumerate(os2):
                    A[row, row + (k-1)*ny] += w * iR2
                for k, w in enumerate(os1):
                    A[row, row + (k-1)*ny] -= w / (R * dR)
            elif i == nx - 2:
                for k, w in enumerate(os2):
                    A[row, row - (k-1)*ny] += w * iR2
                for k, w in enumerate(os1):
                    A[row, row - (k-1)*ny] += w / (R * dR)
            else:
                for k, w in enumerate(c2):
                    A[row, row + (k-2)*ny] += w * iR2
                for k, w in enumerate(c1):
                    A[row, row + (k-2)*ny] -= w / (R * dR)

    for i in range(nx):
        for j in [0, ny-1]:
            A[i*ny+j, i*ny+j] = 1.0
    for i in [0, nx-1]:
        for j in range(ny):
            A[i*ny+j, i*ny+j] = 1.0

    return A.tocsr()


# Alias
gs_sparse = gs_sparse_2nd


# ─────────────────────────────────────────────────────────────────────────────
#  Solver factory  (always CPU internally; result moved to active backend)
# ─────────────────────────────────────────────────────────────────────────────

def make_solver(Rmin, Rmax, Zmin, Zmax, nx, ny, order=2, method='auto'):
    """
    Build a callable  solve(rhs_2d) → psi_2d  on the active backend.

    The sparse matrix and factorisation always live on CPU (scipy).
    rhs_2d is converted to numpy automatically; result is moved to the
    active backend (GPU if set_backend('gpu') was called).

    Parameters
    ----------
    order  : 2 or 4  — FDM order
    method : 'auto' | 'lu' | 'amg'
             'auto' → LU for nx*ny≤16384, AMG otherwise
    """
    if order == 4:
        A = gs_sparse_4th(Rmin, Rmax, Zmin, Zmax, nx, ny)
    else:
        A = gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, nx, ny)

    if method == 'auto':
        method = 'lu' if nx * ny <= 16384 else 'amg'

    _amg = False
    if method == 'amg':
        try:
            import pyamg
            _amg = True
        except ImportError:
            import warnings
            warnings.warn("pyamg not installed; falling back to LU solver. "
                          "Install: pip install pyamg")

    if _amg:
        # Build AMG on the positive-definite interior sub-block only.
        # The full GS matrix has identity rows at the boundary (Dirichlet)
        # which make it indefinite and unsuitable for standard AMG+CG.
        _int = np.array(
            [i*ny + j for i in range(nx) for j in range(ny)
             if 0 < i < nx-1 and 0 < j < ny-1], dtype=int)
        _bn  = np.array(
            [i*ny + j for i in range(nx) for j in range(ny)
             if not (0 < i < nx-1 and 0 < j < ny-1)], dtype=int)
        _Af  = A.tocsr()
        _Ai  = _Af[_int][:, _int]   # interior–interior block (pos-def)
        _Aib = _Af[_int][:, _bn]    # interior–boundary coupling

        try:
            _ml = pyamg.smoothed_aggregation_solver(_Ai.tocsr())
        except Exception:
            _ml = pyamg.ruge_stuben_solver(_Ai.tocsr())

        def solve(rhs_2d):
            rhs_np = to_numpy(rhs_2d).ravel().copy()
            xb     = rhs_np[_bn]                   # boundary values (Dirichlet)
            b_int  = rhs_np[_int] - _Aib @ xb      # corrected interior RHS
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_int = _ml.solve(b_int, tol=1e-10, accel='fgmres')
            out = rhs_np.copy()
            out[_int] = x_int
            return to_backend(out.reshape(nx, ny))

    else:
        from scipy.sparse.linalg import factorized
        _lu = factorized(A.tocsc())

        def solve(rhs_2d):
            rhs_np = to_numpy(rhs_2d).ravel()
            return to_backend(_lu(rhs_np).reshape(nx, ny))

    solve._order  = order
    solve._method = 'amg' if _amg else 'lu'
    solve._nx, solve._ny = nx, ny
    return solve
