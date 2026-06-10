"""
gspack v2.0 — 03-fixedboundary-largegrid
==========================================
Fixed-boundary GS solve with ψ outside the plasma computed via
Green's function volume integral.

Two ψ fields are available:
  1. FDM solver (eq.psi()):  ψ=0 enforced on the D-shape contour
     (Dirichlet BC on the rectangular domain).  Used for internal
     plasma diagnostics (q, βp, etc.).
  2. Green volume integral (eq.psi_on_grid()): free-space ψ from
     the plasma current.  Satisfies Laplace in the vacuum exterior
     with natural BC (ψ→0 at infinity).  Physically correct in the
     external region; ψ on the D-shape is approximately (not exactly)
     zero due to the different BC treatment.
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
METHOD       = "lu"
ANDERSON_M   = 5
BACKEND      = "cpu"

# D-shaped LCFS
R0, a0       = 1.0, 0.5
KAPPA, DELTA = 1.6, 0.33

# Profile
Ip_target    = 2e5
betap_target = 0.8
fvac         = 1.0

# Large-grid domain (vessel scale)
R_vmin, R_vmax = 0.1, 2.5
Z_vmin, Z_vmax = -1.2, 1.2
NX_v, NY_v     = 97, 97

bk.set_backend(BACKEND)
script_dir = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
#  Step 1:  Solve fixed-boundary GS inside D-shape (same as 02)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("  gspack v2.0  —  03-fixedboundary-largegrid")
print("=" * 64)

print("\n  ── Step 1: Solve GS inside D-shape LCFS ──")
eq = FixedBoundaryEquilibrium(
    R0=R0, a=a0, kappa=KAPPA, delta=DELTA,
    Rmin=0.2, Rmax=1.8, Zmin=-0.8, Zmax=0.8,
    nx=NX, ny=NY, order=ORDER, method=METHOD)

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
print(f"  ψ_a = {eq.psi_axis:.4f} Wb/rad")

# ═══════════════════════════════════════════════════════════════════════════
#  Step 2:  Extend ψ to a large grid via Green's function volume integral
# ═══════════════════════════════════════════════════════════════════════════
print("\n  ── Step 2: Green's function volume integral on large grid ──")
print(f"  Large grid: {NX_v}×{NY_v}  "
      f"[{R_vmin}, {R_vmax}] × [{Z_vmin}, {Z_vmax}]")
print(f"  Source points inside D-shape: {eq.plasma_mask.sum():.0f}")

R1d_v = np.linspace(R_vmin, R_vmax, NX_v)
Z1d_v = np.linspace(Z_vmin, Z_vmax, NY_v)
R_v, Z_v = np.meshgrid(R1d_v, Z1d_v, indexing='ij')

psi_v = eq.psi_on_grid(R_v, Z_v)

# ═══════════════════════════════════════════════════════════════════════════
#  Step 3:  Visualise
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: FDM solution on small domain (ψ=0 on D-shape)
ax = axes[0]
cf = ax.contourf(eq.R, eq.Z, eq.psi(), levels=30, cmap="RdYlBu_r")
ax.contour(eq.R, eq.Z, eq.psi(),
           levels=np.linspace(eq.psi_bndry, eq.psi_axis, 14),
           colors="white", linewidths=0.6)
ax.plot(eq.R_lcfs, eq.Z_lcfs, 'g-', lw=2, label='D-shape LCFS')
ax.plot(eq.magneticAxis()[0], eq.magneticAxis()[1], 'k+', ms=12, mew=2)
ax.set_xlabel('R [m]'); ax.set_ylabel('Z [m]')
ax.set_title(f'FDM solver  (ψ=0 on D-shape)\nψ_axis={eq.psi_axis:.4f}')
ax.set_aspect('equal'); ax.legend(fontsize=8)
plt.colorbar(cf, ax=ax)

# Panel 2: Green volume integral on large grid (free-space ψ)
ax = axes[1]
levels = np.linspace(psi_v.min(), psi_v.max(), 50)
cf2 = ax.contourf(R_v, Z_v, psi_v, levels=levels, cmap="RdYlBu_r")
ax.contour(R_v, Z_v, psi_v,
           levels=np.linspace(psi_v.min(), psi_v.max(), 14),
           colors="white", linewidths=0.6, alpha=0.4)
ax.plot(eq.R_lcfs, eq.Z_lcfs, 'g-', lw=2, label='D-shape')
ax.set_xlim(R_vmin, R_vmax); ax.set_ylim(Z_vmin, Z_vmax)
ax.set_xlabel('R [m]'); ax.set_ylabel('Z [m]')
ax.set_title(f'Green volume integral (free-space)\nψ on large grid')
ax.set_aspect('equal'); ax.legend(fontsize=8)
plt.colorbar(cf2, ax=ax)

# Panel 3: Green integral zoomed to small domain (with FDM contours overlaid)
ax = axes[2]
psi_green_detail = eq.psi_on_grid(eq.R, eq.Z)
cf3 = ax.contourf(eq.R, eq.Z, psi_green_detail, levels=30, cmap="RdYlBu_r")
cs_fdm = ax.contour(eq.R, eq.Z, eq.psi(),
                     levels=np.linspace(eq.psi_bndry, eq.psi_axis, 10),
                     colors='k', linewidths=1.0, linestyles='--', alpha=0.6)
ax.clabel(cs_fdm, inline=1, fontsize=8, fmt='%.2e')
ax.plot(eq.R_lcfs, eq.Z_lcfs, 'g-', lw=2, label='D-shape')
ax.set_xlabel('R [m]'); ax.set_ylabel('Z [m]')
ax.set_title('Green integral (filled) + FDM contours (dashed)')
ax.set_aspect('equal'); ax.legend(fontsize=8)
plt.colorbar(cf3, ax=ax)

plt.tight_layout()
plot_path = os.path.join(script_dir, "03_fixedboundary_largegrid.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  Summary diagnostics on the large grid
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n  ── Large-grid diagnostics ──")
print(f"  ψ on large grid: [{psi_v.min():.4e}, {psi_v.max():.4e}] Wb/rad")
# Br/Bz from large grid psi
from scipy.interpolate import RectBivariateSpline
f_psi_v = RectBivariateSpline(R1d_v, Z1d_v, psi_v)
Br_v = -f_psi_v(R1d_v, Z1d_v, dy=1, grid=True) / R_v
Bz_v =  f_psi_v(R1d_v, Z1d_v, dx=1, grid=True) / R_v
Bpol_v = np.sqrt(Br_v**2 + Bz_v**2)

# Poloidal field at a few representative points
for label, r, z in [("Inboard midplane", 0.15, 0.0),
                     ("Outboard midplane", 2.3, 0.0),
                     ("Top", 1.0, 1.1),
                     ("Bottom", 1.0, -1.1)]:
    b = float(np.sqrt(
        float(f_psi_v(r, z, dy=1))**2 + float(f_psi_v(r, z, dx=1))**2) / r)
    print(f"  |Bpol| at {label:20s} ({r:.2f},{z:.2f}):  {b:.4e} T")

print(f"\n  Plot saved  →  {os.path.basename(plot_path)}")
print("\n" + "=" * 64)
print("  Done.")
print("=" * 64)
