"""
gspack v2.0 — 03-fixedboundary-largegrid
==========================================
Fixed-boundary GS solve with external-region ψ computed via
Green's function volume integral on the entire large grid.

  ψ(R,Z) = ∫∫ G(R,Z; R',Z') · J_φ(R',Z') dR' dZ'

This gives a single, continuous ψ field everywhere (no FDM/Green
mixing → no discontinuity at D-shape).  Satisfies:
  Δ*ψ = -μ₀ R Jφ inside plasma, Δ*ψ = 0 in vacuum, ψ → 0 at ∞.

On the D-shape contour, ψ varies (not exactly constant) because
the Green integral does not enforce a Dirichlet BC.  The constant-ψ
LCFS is enforced in the FDM solve (eq.psi()) for internal diagnostics.
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gspack.equilibrium import FixedBoundaryEquilibrium
from gspack.profiles    import ConstrainBetapIp
from gspack             import picard
import gspack.backend as bk

# ── Config ────────────────────────────────────────────────────────────────────
NX, NY       = 65, 65
ORDER        = 2
ANDERSON_M   = 5
BACKEND      = "cpu"

# D-shaped LCFS
R0, a0       = 1.0, 0.5
KAPPA, DELTA = 1.6, 0.33

# Profile
Ip_target    = 2e5
betap_target = 0.8
fvac         = 1.0

# Large-grid domain (vessel scale) and resolution
R_vmin, R_vmax = 0.1, 2.5
Z_vmin, Z_vmax = -1.2, 1.2
NX_v, NY_v     = 129, 129

bk.set_backend(BACKEND)
script_dir = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
#  Step 1:  Solve fixed-boundary GS inside D-shape
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("  gspack v2.0  —  03-fixedboundary-largegrid")
print("=" * 64)

print("\n  ── Step 1: Solve GS inside D-shape LCFS ──")
eq = FixedBoundaryEquilibrium(
    R0=R0, a=a0, kappa=KAPPA, delta=DELTA,
    Rmin=0.2, Rmax=1.8, Zmin=-0.8, Zmax=0.8,
    nx=NX, ny=NY, order=ORDER)

pro = ConstrainBetapIp(
    betap=betap_target, Ip=Ip_target, fvac=fvac,
    alpha_m=1.0, alpha_n=2.0, Raxis=R0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    picard.solve(eq, pro, constrain=None,
                 maxits=50, rtol=1e-3,
                 anderson_m=ANDERSON_M, convergenceInfo=True)

print(f"  Ip  = {eq.plasmaCurrent():.2e} A  (target {Ip_target:.0e} A)")
print(f"  βp  = {eq.poloidalBeta():.3f}  (target {betap_target})")
print(f"  ψ_a = {eq.psi_axis:.4f}, ψ_bndry = {eq.psi_bndry:.4f} Wb/rad")

# ═══════════════════════════════════════════════════════════════════════════
#  Step 2:  Green integral ψ on large grid
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  ── Step 2: Green integral on large grid ──")
print(f"  Grid: {NX_v}×{NY_v}  [{R_vmin},{R_vmax}]×[{Z_vmin},{Z_vmax}]")

R_v, Z_v, psi_v = eq.psi_on_grid(
    R_vmin, R_vmax, Z_vmin, Z_vmax, NX_v, NY_v,
    order=ORDER, method='lu')

# ── Verify D-shape continuity ───────────────────────────────────────────
from scipy.interpolate import RectBivariateSpline
f_v = RectBivariateSpline(R_v[:,0], Z_v[0,:], psi_v)
psi_on_ds = np.array([float(f_v(r, z)) for r, z in zip(eq.R_lcfs, eq.Z_lcfs)])
print(f"  ψ on D-shape (Green): mean={psi_on_ds.mean():.4e}, "
      f"range=[{psi_on_ds.min():.4e}, {psi_on_ds.max():.4e}]")
print(f"  ψ_bndry (FDM) = {eq.psi_bndry:.4f}")
print(f"  Δψ_spread = {psi_on_ds.max() - psi_on_ds.min():.2e} (Green integral spread on LCFS)")

# ═══════════════════════════════════════════════════════════════════════════
#  Step 3:  Visualise
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: FDM solution on small domain
ax = axes[0]
cf = ax.contourf(eq.R, eq.Z, eq.psi(), levels=30, cmap="RdYlBu_r")
ax.contour(eq.R, eq.Z, eq.psi(),
           levels=np.linspace(eq.psi_bndry, eq.psi_axis, 14),
           colors="white", linewidths=0.6)
ax.plot(eq.R_lcfs, eq.Z_lcfs, 'g-', lw=2, label='D-shape LCFS')
ax.plot(eq.magneticAxis()[0], eq.magneticAxis()[1], 'k+', ms=12, mew=2)
ax.set_title(f'FDM on small domain\nψ_axis={eq.psi_axis:.4f}')
ax.set_aspect('equal'); ax.legend(fontsize=8)
plt.colorbar(cf, ax=ax)

# Panel 2: Laplace-extended solution on large grid
ax = axes[1]
lvls = np.linspace(psi_v.min(), psi_v.max(), 50)
cf2 = ax.contourf(R_v, Z_v, psi_v, levels=lvls, cmap="RdYlBu_r")
ax.contour(R_v, Z_v, psi_v,
           levels=np.linspace(eq.psi_bndry, eq.psi_axis, 14),
           colors="white", linewidths=0.6, alpha=0.4)
ax.plot(eq.R_lcfs, eq.Z_lcfs, 'g-', lw=2, label='D-shape')
ax.set_xlim(R_vmin, R_vmax); ax.set_ylim(Z_vmin, Z_vmax)
ax.set_title('Green integral on large grid\n(Δ*ψ=0 in vacuum, ψ→0 at ∞)')
ax.set_aspect('equal'); ax.legend(fontsize=8)
plt.colorbar(cf2, ax=ax)

# Panel 3: Zoom — continuity at D-shape boundary
ax = axes[2]
# Extract a 1-D cut along Z=0
iz0 = np.argmin(np.abs(Z_v[0,:]))
R1d_v = R_v[:, iz0]
psi_v_cut = psi_v[:, iz0]
# FDM solution for comparison
f_fdm = RectBivariateSpline(eq.R[:,0], eq.Z[0,:], eq.psi())
Z0 = Z_v[0, iz0]
psi_fdm_cut = f_fdm(R1d_v, Z0, grid=False)

ax.plot(R1d_v, psi_v_cut, 'b-', lw=2, label='Green integral + FDM')
ax.plot(R1d_v, psi_fdm_cut, 'r--', lw=2, label='FDM solver')
ax.axvline(eq.R_lcfs.min(), color='gray', ls=':', label='D-shape limits')
ax.axvline(eq.R_lcfs.max(), color='gray', ls=':')
ax.axhline(eq.psi_bndry, color='green', ls='--', lw=1, alpha=0.7)
ax.set_xlabel('R [m]'); ax.set_ylabel('ψ [Wb/rad]')
ax.set_title(f'ψ(R, Z={Z0:.2f})  —  Z=0 midplane cut')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(R_vmin, R_vmax)

plt.tight_layout()
plot_path = os.path.join(script_dir, "03_fixedboundary_largegrid.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostics on large grid
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  ── Large-grid diagnostics ──")
print(f"  ψ range: [{psi_v.min():.4e}, {psi_v.max():.4e}] Wb/rad")
f_v = RectBivariateSpline(R_v[:,0], Z_v[0,:], psi_v)
Br_v = -f_v(R_v[:,0], Z_v[0,:], dy=1, grid=True) / R_v
Bz_v =  f_v(R_v[:,0], Z_v[0,:], dx=1, grid=True) / R_v

for label, r, z in [("Inboard (R=0.15)", 0.15, 0.0),
                     ("Outboard (R=2.3)", 2.3, 0.0),
                     ("Top (Z=1.1)", 1.0, 1.1)]:
    b = float(np.sqrt(
        float(f_v(r, z, dy=1))**2 + float(f_v(r, z, dx=1))**2) / r)
    print(f"  |Bpol| at {label:20s}  {b:.4e} T")

print(f"\n  Plot saved  →  {os.path.basename(plot_path)}")
print("\n" + "=" * 64)
print("  Done.")
print("=" * 64)
