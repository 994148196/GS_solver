"""
gspack.critical
===============
Find O-points (magnetic axis) and X-points in a 2-D psi field,
and construct a core mask that is 1 inside the plasma and 0 outside.

Algorithm
---------
1. Compute |∇ψ|² on the grid via spline differentiation.
2. Find ALL local minima of |∇ψ|² as candidate field-null points
   (using size-3 minimum filter to catch even weak minima).
3. Refine each candidate with Newton iterations on Br=Bz=0.
4. Classify by S = d²ψ/dR² d²ψ/dZ² - (d²ψ/dRdZ)²:
     S > 0  →  O-point (local extremum of ψ)
     S < 0  →  X-point (saddle)

Reference: FreeGS critical.py (Dudson 2016).
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import minimum_filter


def find_critical(R, Z, psi):
    """
    Find O-points and X-points in psi(R, Z).

    Parameters
    ----------
    R, Z : 2-D arrays (nx, ny) — meshgrid (indexing='ij')
    psi  : 2-D array  (nx, ny)

    Returns
    -------
    opoints : list of (R, Z, psi)  — O-points
    xpoints : list of (R, Z, psi)  — X-points
    """
    R1d = R[:, 0]
    Z1d = Z[0, :]
    nx, ny = psi.shape

    # Bicubic spline
    f = RectBivariateSpline(R1d, Z1d, psi)

    # |∇ψ|² on the grid
    dpsidR = f(R1d, Z1d, dx=1)   # (nx, ny)
    dpsidZ = f(R1d, Z1d, dy=1)   # (nx, ny)
    Bp2    = dpsidR**2 + dpsidZ**2

    dR = R1d[1] - R1d[0]
    dZ = Z1d[1] - Z1d[0]

    # Find ALL local minima with size-3 filter (catches weak O-points)
    local_min = minimum_filter(Bp2, size=3)
    # Accept all local minima that are below 1% of the global max
    # (generous threshold to not miss the O-point even on coarse grids)
    tol_sq = max(1e-2 * Bp2.max(), 1e-30)
    candidate_mask = (Bp2 == local_min) & (Bp2 < tol_sq)
    seeds_ij = list(zip(*np.where(candidate_mask)))

    # Also always seed from the global minimum (catches the O-point)
    gmin_idx = np.unravel_index(Bp2.argmin(), Bp2.shape)
    all_seeds = set(seeds_ij)
    all_seeds.add(gmin_idx)
    seeds_ij = list(all_seeds)

    opoints, xpoints = [], []
    seen = []

    for i0, j0 in seeds_ij:
        i0, j0 = int(i0), int(j0)
        # Skip boundary points
        if i0 < 1 or i0 > nx - 2 or j0 < 1 or j0 > ny - 2:
            continue

        R0 = R1d[i0]
        Z0 = Z1d[j0]

        # Newton–Raphson refinement
        R1, Z1 = _newton_null(f, R0, Z0, dR, dZ)
        if R1 is None:
            continue
        if not (R1d[0] < R1 < R1d[-1] and Z1d[0] < Z1 < Z1d[-1]):
            continue

        # De-duplicate (within 3 grid cells)
        duplicate = False
        for Rp, Zp, _ in seen:
            if (R1 - Rp)**2 + (Z1 - Zp)**2 < (3 * max(dR, dZ))**2:
                duplicate = True
                break
        if duplicate:
            continue

        psi_val = float(f(R1, Z1).flat[0])
        seen.append((R1, Z1, psi_val))

        # Classify by S = d²ψ/dR² * d²ψ/dZ² - (d²ψ/dRdZ)²
        d2R2 = float(f(R1, Z1, dx=2).flat[0])
        d2Z2 = float(f(R1, Z1, dy=2).flat[0])
        d2RZ = float(f(R1, Z1, dx=1, dy=1).flat[0])
        S    = d2R2 * d2Z2 - d2RZ**2

        if S > 0:
            opoints.append((R1, Z1, psi_val))
        else:
            xpoints.append((R1, Z1, psi_val))

    # Sort O-points by distance from domain centre (FreeGS convention)
    if opoints:
        Rmid = 0.5 * (R1d[0] + R1d[-1])
        Zmid = 0.5 * (Z1d[0] + Z1d[-1])
        opoints.sort(key=lambda p: (p[0] - Rmid)**2 + (p[1] - Zmid)**2)

    # Filter X-points: keep only those with monotonic psi from O-point (FreeGS logic)
    if opoints and xpoints:
        Ro, Zo, Po = opoints[0]
        xpt_keep = []
        for xpt in xpoints:
            Rx, Zx, Px = xpt
            rline = np.linspace(Ro, Rx, 50)
            zline = np.linspace(Zo, Zx, 50)
            pline = f(rline, zline, grid=False)
            if Px < Po:
                pline = -pline  # normalise so max at X-point end
            maxp = pline.max()
            if (maxp - pline[-1]) / (maxp - pline[0] + 1e-30) > 0.001:
                continue  # non-monotonic: discard
            ind = pline.argmin()
            if (rline[ind] - Ro)**2 + (zline[ind] - Zo)**2 > 1e-4:
                continue  # minimum not near O-point: discard
            xpt_keep.append(xpt)
        xpoints = xpt_keep

    # Sort X-points by psi distance from primary O-point
    if opoints and xpoints:
        psi_ax = opoints[0][2]
        xpoints.sort(key=lambda p: (p[2] - psi_ax)**2)

    return opoints, xpoints


def _newton_null(f, R0, Z0, dR, dZ, maxiter=50, tol=1e-10):
    """Newton iteration to find Br=Bz=0 starting from (R0, Z0)."""
    R1, Z1 = float(R0), float(Z0)

    for _ in range(maxiter):
        Br = -float(f(R1, Z1, dy=1).flat[0]) / R1
        Bz =  float(f(R1, Z1, dx=1).flat[0]) / R1

        if Br**2 + Bz**2 < tol:
            return R1, Z1

        # Jacobian of (Br, Bz) with respect to (R, Z)
        J00 = -float(f(R1, Z1, dx=1, dy=1).flat[0]) / R1 - Bz / R1
        J01 = -float(f(R1, Z1, dy=2).flat[0])        / R1
        J10 =  float(f(R1, Z1, dx=2).flat[0])        / R1 - Bz / R1
        J11 =  float(f(R1, Z1, dx=1, dy=1).flat[0])  / R1

        det = J00 * J11 - J01 * J10
        if abs(det) < 1e-30:
            break

        dR_step = -(J11 * Br - J01 * Bz) / det
        dZ_step = -(-J10 * Br + J00 * Bz) / det

        # Limit step size for stability
        step = np.sqrt(dR_step**2 + dZ_step**2)
        max_step = 0.5 * max(dR, dZ) * 10
        if step > max_step:
            dR_step *= max_step / step
            dZ_step *= max_step / step

        R1 += dR_step
        Z1 += dZ_step

    # Final convergence check
    Br = -float(f(R1, Z1, dy=1).flat[0]) / R1
    Bz =  float(f(R1, Z1, dx=1).flat[0]) / R1
    if Br**2 + Bz**2 < 1e-6:
        return R1, Z1
    return None, None


def core_mask(R, Z, psi, opoints, xpoints, psi_bndry=None):
    """
    Return 2-D mask: 1 inside plasma core, 0 outside.

    Uses the same flood-fill algorithm as FreeGS:
    - Starts at the O-point grid cell
    - Expands to neighbours where psiN < 1
    - X-point cells are blocked during fill, then re-evaluated after

    This correctly handles the saddle-point topology at X-points.
    """
    if not opoints:
        return None

    nx, ny = psi.shape
    psi_axis = opoints[0][2]

    if psi_bndry is None:
        if xpoints:
            psi_bndry = xpoints[0][2]
        else:
            return None

    denom = psi_bndry - psi_axis
    if abs(denom) < 1e-30:
        return None

    psin = (psi - psi_axis) / denom

    R1d = R[:, 0]
    Z1d = Z[0, :]

    mask = np.zeros(psi.shape)

    # Block X-point cells to prevent flood-fill leakage through saddle point
    xpt_inds = []
    for xp in xpoints:
        ix = int(np.argmin(np.abs(R1d - xp[0])))
        jx = int(np.argmin(np.abs(Z1d - xp[1])))
        xpt_inds.append((ix, jx))
        for ii in np.clip([ix-1, ix, ix+1], 0, nx-1):
            for jj in np.clip([jx-1, jx, jx+1], 0, ny-1):
                mask[ii, jj] = 2.0   # blocked

    # Seed from O-point
    Ro, Zo = opoints[0][0], opoints[0][1]
    rind = int(np.argmin(np.abs(R1d - Ro)))
    zind = int(np.argmin(np.abs(Z1d - Zo)))

    stack = [(rind, zind)]

    while stack:
        i, j = stack.pop()

        # Check left neighbour
        if j > 0 and psin[i, j-1] < 1.0 and mask[i, j-1] < 0.5:
            stack.append((i, j-1))

        # Scan row to the right
        while True:
            mask[i, j] = 1.0

            if i < nx-1 and psin[i+1, j] < 1.0 and mask[i+1, j] < 0.5:
                stack.append((i+1, j))
            if i > 0   and psin[i-1, j] < 1.0 and mask[i-1, j] < 0.5:
                stack.append((i-1, j))

            if j == ny-1:
                break
            if psin[i, j+1] >= 1.0 or mask[i, j+1] > 0.5:
                break
            j += 1

    # Re-evaluate X-point cells
    for ix, jx in xpt_inds:
        for ii in np.clip([ix-1, ix, ix+1], 0, nx-1):
            for jj in np.clip([jx-1, jx, jx+1], 0, ny-1):
                mask[ii, jj] = 1.0 if psin[ii, jj] < 1.0 else 0.0

    return mask


def update_psi_boundary(R, Z, psi):
    """
    Determine psi_axis, psi_bndry, opoints, xpoints from psi.

    Returns (psi_axis, psi_bndry, opoints, xpoints).
    """
    opoints, xpoints = find_critical(R, Z, psi)

    if not opoints:
        return None, None, opoints, xpoints

    psi_axis = opoints[0][2]

    if xpoints:
        # psi_bndry = X-point psi closest to plasma boundary
        # For Ip>0 convention: psi_axis > psi_bndry
        # Pick the X-point with psi closest to (but less than) psi_axis
        psi_xpt_vals = [xp[2] for xp in xpoints]
        if psi_axis > np.mean(psi_xpt_vals):
            # Ip > 0: axis is maximum, boundary is the highest xpt psi below axis
            candidates = [v for v in psi_xpt_vals if v < psi_axis]
            psi_bndry  = max(candidates) if candidates else psi_xpt_vals[0]
        else:
            # Ip < 0
            candidates = [v for v in psi_xpt_vals if v > psi_axis]
            psi_bndry  = min(candidates) if candidates else psi_xpt_vals[0]
    else:
        bndry_vals = np.concatenate([
            psi[0, :], psi[-1, :], psi[:, 0], psi[:, -1]])
        psi_bndry = float(np.mean(bndry_vals))

    return psi_axis, psi_bndry, opoints, xpoints
