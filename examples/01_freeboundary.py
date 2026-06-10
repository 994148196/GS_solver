"""
gspack v2.0 — 01-freeboundary
==============================
Reproduces FreeGS 01-freeboundary.py, demonstrating all v2 features:
  • Anderson mixing acceleration
  • 2nd or 4th-order FDM
  • G-EQDSK output
  • HDF5 output
  • MHD stability indicators
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gspack.machine     import TestTokamak
from gspack.equilibrium import Equilibrium
from gspack.profiles    import ConstrainPaxisIp
from gspack.control     import constrain
from gspack             import picard, geqdsk, io, diagnostics
import gspack.backend as bk

# ── Config ────────────────────────────────────────────────────────────────────
NX, NY      = 65, 65
ORDER       = 2          # 2 (fast) or 4 (accurate)
METHOD      = "lu"       # "lu", "amg", "auto"
ANDERSON_M  = 5          # 0 = plain Picard
BACKEND     = "gpu"      # "cpu" or "gpu"

bk.set_backend(BACKEND)
script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Setup ─────────────────────────────────────────────────────────────────────
tok = TestTokamak()
eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                  nx=NX, ny=NY, order=ORDER, method=METHOD)
pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
con = constrain(
    xpoints=[(1.1, -0.6), (1.1, 0.6)],
    isoflux=[(1.1,-0.6, 1.1,0.6),
             (1.1,-0.6, 1.7,0.0),
             (1.1, 0.6, 1.7,0.0)],
    gamma=1e-12)

# ── Solve ─────────────────────────────────────────────────────────────────────
print("=" * 64)
print("  gspack v2.0  —  01-freeboundary")
print("=" * 64)
print(f"\n  Backend : {bk.get_backend().upper()}")
print(f"  Grid    : {NX}×{NY}  FDM order={ORDER}  solver={METHOD}")
print(f"  Anderson mixing m={ANDERSON_M}")
print(f"\n  Machine: TestTokamak")
for name, coil in tok.coils:
    print(f"    {name}  {coil}")
print()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    errs = picard.solve(eq, pro, con, maxits=50, rtol=1e-3,
                        anderson_m=ANDERSON_M, convergenceInfo=True)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  RESULTS")
print("=" * 64)
print(f"\n  Plasma current Ip  = {eq.plasmaCurrent():.6e} A")
print(f"  psi axis           = {eq.psi_axis:.6f} Wb/rad")
print(f"  psi boundary       = {eq.psi_bndry:.6f} Wb/rad")
mag = eq.magneticAxis()
print(f"\n  Magnetic axis: R = {mag[0]:.4f} m,  Z = {mag[1]:.2e} m")
geo = eq.geometricAxis()
print(f"  Geometric axis: R = {geo[0]:.4f} m")
sh = eq.shafranovShift()
print(f"  Shafranov shift dR = {sh[0]*100:.3f} cm")
Ri, Ro = eq.innerOuterSeparatrix()
print(f"\n  Minor radius a  = {eq.minorRadius():.4f} m")
print(f"  Elongation κ    = {eq.elongation():.4f}")
print(f"  Triangularity δ = {eq.triangularity():.4f}")
print(f"  Aspect ratio R/a= {eq.aspectRatio():.4f}")
print(f"  Plasma volume V = {eq.plasmaVolume():.4f} m³")
print(f"\n  Separatrix:  Rinner={Ri:.4f} m,  Router={Ro:.4f} m")

psiN_q = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
q_vals = eq.q(psiN_q)
print(f"\n  Safety factor q(ψN):")
for pN, q in zip(psiN_q, q_vals):
    print(f"    q({pN:.2f}) = {q:.4f}")

print(f"\n  PF Coil currents:")
print("  " + "=" * 28)
for name, coil in tok.coils:
    print(f"  {name:5s}  {coil.current:+.1f} A")
print("  " + "=" * 28)

# ── MHD stability ─────────────────────────────────────────────────────────────
print(f"\n  MHD stability indicators:")
stab = diagnostics.mhd_stability(eq)
print(f"    q axis = {stab['q_axis']:.3f}")
print(f"    q95    = {stab['q_95']:.3f}")
print(f"    Greenwald limit = {stab['Greenwald_density_limit_1e20']:.2f} e20 m-3")
print(f"    betaN = {stab['betaN']:.4f}  Troyon margin = {stab['Troyon_margin']:.3f}")
print(f"    li = {stab['li']:.4f}")
for qv in [1, 2, 3]:
    pN = stab.get(f'q={qv} surface psiN', float('nan'))
    tag = f'{pN:.3f}' if np.isfinite(pN) else 'outside plasma'
    print(f'    q={qv} surface at psiN = {tag}')
print(f"    li = {stab['li']:.4f}")

# ── G-EQDSK output ────────────────────────────────────────────────────────────
geqdsk_path = os.path.join(script_dir, "..", "01_freeboundary.geqdsk")
geqdsk.write(eq, geqdsk_path)

# ── HDF5 output ───────────────────────────────────────────────────────────────
h5_path = os.path.join(script_dir, "..", "01_freeboundary.h5")
io.save(eq, h5_path)

# ── Plot ─────────────────────────────────────────────────────────────────────
psi2d = eq.psi()
sep   = eq.separatrix(npoints=360)
mag   = eq.magneticAxis()

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle(
    f"gspack v2.0  ·  {NX}×{NY}  ·  FDM order={ORDER}  ·  Anderson m={ANDERSON_M}",
    fontsize=12)

# Flux surfaces
ax = axes[0]
cf = ax.contourf(eq.R, eq.Z, psi2d, levels=30, cmap="RdYlBu_r")
ax.contour(eq.R, eq.Z, psi2d,
           levels=np.linspace(eq.psi_bndry, eq.psi_axis, 14),
           colors="white", linewidths=0.6)
ax.contour(eq.R, eq.Z, psi2d, levels=[eq.psi_bndry],
           colors=["lime"], linewidths=2.0)
ax.plot(sep[:, 0], sep[:, 1], "g-", lw=1.5, label="LCFS")
ax.plot(mag[0], mag[1], "k+", ms=12, mew=2, label="O-point")
for xp in eq._xpoints[:2]:
    ax.plot(xp[0], xp[1], "rx", ms=10, mew=2)
for _, coil in tok.coils:
    ax.add_patch(plt.Circle((coil.R, coil.Z), 0.04,
                             color="orange", zorder=5))
ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_title("ψ(R,Z)")
ax.set_aspect("equal"); ax.legend(fontsize=8)
plt.colorbar(cf, ax=ax)

# q profile
ax2 = axes[1]
psiN_fine = np.linspace(0.02, 0.97, 80)
q_fine    = eq.q(psiN_fine)
ax2.plot(psiN_fine, q_fine, "b-", lw=2)
for qv in [1.0, 2.0, 3.0]:
    ax2.axhline(qv, color="gray", ls="--", lw=0.8, alpha=0.6)
ax2.set_xlabel("ψN"); ax2.set_ylabel("q(ψN)")
ax2.set_title("Safety factor")
ax2.set_xlim(0, 1); ax2.grid(True, alpha=0.3)

# Convergence
ax3 = axes[2]
if errs is not None:
    iters = np.arange(1, len(errs[1]) + 1)
    ax3.semilogy(iters, errs[1], "b-o", ms=4, lw=1.5, label="rel|Δψ|")
    ax3.axhline(1e-3, color="red", ls="--", lw=1, label="rtol=1e-3")
    ax3.set_xlabel("Iteration"); ax3.set_ylabel("Relative change")
    ax3.set_title(f"Convergence  (Anderson m={ANDERSON_M})")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(script_dir, "01_freeboundary_v2.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved  →  {os.path.basename(plot_path)}")
print("\n" + "=" * 64)
print("  Done.")
print("=" * 64)
