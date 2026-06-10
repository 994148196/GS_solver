"""
gspack v2.0 — 02-fixedboundary
===============================
Fixed-boundary Grad–Shafranov solve with prescribed D-shaped LCFS.

The plasma boundary is defined by (R₀, a, κ, δ) — no external coils.

Current profile: Jeon (2015) Eq.(5)
    J_φ = λ [β₀ R/R₀ + (1-β₀)R₀/R] (1 - ψ̂^{α_m})^{α_n}
with λ, β₀ constrained by total current I_p and poloidal beta β_p
(Jeon Eqs. 13a, 13b).

The boundary value ψ_bndry on the D-shaped LCFS is computed self-
consistently from the Green's function volume integral of the plasma
current.  Each Picard iteration:
  1. Computes J_φ from the current ψ distribution
  2. Computes ψ_green = ∫G·J_φ dS on the D-shape contour → ψ_bndry
  3. Solves GS inside D-shape with ψ = ψ_bndry Dirichlet BC

After convergence, the Green integral of the converged Jtor gives
ψ ≈ ψ_bndry on the D-shape (self-consistent LCFS flux surface).
Use eq.psi_on_grid() to extend ψ to any external grid via the
Green volume integral.
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gspack.equilibrium import FixedBoundaryEquilibrium
from gspack.profiles    import ConstrainBetapIp
from gspack             import picard, diagnostics
import gspack.backend as bk

# ── Config ────────────────────────────────────────────────────────────────────
NX, NY       = 65, 65
ORDER        = 2          # 2 (fast) or 4 (accurate)
METHOD       = "lu"       # "lu", "amg", "auto"
ANDERSON_M   = 5          # 0 = plain Picard
BACKEND      = "cpu"      # "cpu" or "gpu"

# D-shaped LCFS parameters (Cerfon–Solovev)
R0          = 1.0         # major radius [m]
a           = 0.5         # minor radius [m]
KAPPA       = 1.6         # elongation
DELTA       = 0.33        # triangularity

# Plasma profile parameters
Ip_target   = 2e5         # total plasma current [A]
betap_target = 0.8        # poloidal beta
fvac        = 1.0         # vacuum R·Bφ [T·m]

bk.set_backend(BACKEND)
script_dir = os.path.dirname(os.path.abspath(__file__))

# ── Setup ─────────────────────────────────────────────────────────────────────
eq = FixedBoundaryEquilibrium(
    R0=R0, a=a, kappa=KAPPA, delta=DELTA,
    Rmin=0.2, Rmax=1.8, Zmin=-0.8, Zmax=0.8,
    nx=NX, ny=NY, order=ORDER, method=METHOD)

pro = ConstrainBetapIp(
    betap=betap_target, Ip=Ip_target, fvac=fvac,
    alpha_m=1.0, alpha_n=2.0, Raxis=R0)

# ── Solve ─────────────────────────────────────────────────────────────────────
print("=" * 64)
print("  gspack v2.0  —  02-fixedboundary")
print("=" * 64)
print(f"\n  Backend : {bk.get_backend().upper()}")
print(f"  Grid    : {NX}×{NY}  FDM order={ORDER}  solver={METHOD}")
print(f"  Anderson mixing m={ANDERSON_M}")
print(f"\n  LCFS:  R₀={R0} m  a={a} m  κ={KAPPA}  δ={DELTA}")
print(f"  Ip    = {Ip_target:.0f} A")
print(f"  beta_p = {betap_target:.2f}")
print()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    errs = picard.solve(eq, pro, constrain=None,
                        maxits=50, rtol=1e-3,
                        anderson_m=ANDERSON_M, convergenceInfo=True)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  RESULTS")
print("=" * 64)
print(f"\n  Plasma current Ip  = {eq.plasmaCurrent():.6e} A  (target {Ip_target:.0e} A)")
print(f"  Poloidal beta βp   = {eq.poloidalBeta():.4f}  (target {betap_target})")
print(f"  psi axis           = {eq.psi_axis:.6f} Wb/rad")
print(f"  psi boundary       = {eq.psi_bndry:.6f} Wb/rad  (Green integral)")
print(f"  ψ₀ = ψ_bndry − ψ_axis = {eq.psi_bndry - eq.psi_axis:.4f} Wb/rad")
# Self-consistency check
idx_i, idx_j = np.where(eq.plasma_mask)
from gspack.boundary import _green_matrix_np
_Gds = _green_matrix_np(eq.R_lcfs, eq.Z_lcfs, eq.R[idx_i, idx_j], eq.Z[idx_i, idx_j])
psi_ds = _Gds @ (eq._Jtor[idx_i, idx_j] * eq.dR * eq.dZ)
print(f"  Self-consistency:  ⟨ψ_green⟩_LCFS = {psi_ds.mean():.4f}")
print(f"                     max|ψ_green − ψ_bndry| = {np.abs(psi_ds - eq.psi_bndry).max():.2e}")
mag = eq.magneticAxis()
print(f"\n  Magnetic axis: R = {mag[0]:.4f} m,  Z = {mag[1]:.2e} m")
geo = eq.geometricAxis()
print(f"  Geometric axis: R = {geo[0]:.4f} m")
sh = eq.shafranovShift()
print(f"  Shafranov shift dR = {sh[0]*100:.3f} cm")
Ri, Ro = eq.innerOuterSeparatrix()
print(f"\n  Minor radius a  = {eq.minorRadius():.4f} m")
print(f"  Elongation κ    = {eq.elongation():.4f}  (target {KAPPA})")
print(f"  Triangularity δ = {eq.triangularity():.4f}  (target {DELTA})")
print(f"  Aspect ratio R/a= {eq.aspectRatio():.4f}")
print(f"  Plasma volume V = {eq.plasmaVolume():.4f} m³")

psiN_q = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
q_vals = eq.q(psiN_q)
print(f"\n  Safety factor q(ψN):")
for pN, q in zip(psiN_q, q_vals):
    print(f"    q({pN:.2f}) = {q:.4f}")

# ── MHD stability ─────────────────────────────────────────────────────────────
print(f"\n  MHD stability indicators:")
stab = diagnostics.mhd_stability(eq)
print(f"    q axis = {stab['q_axis']:.3f}")
print(f"    q95    = {stab['q_95']:.3f}")
print(f"    Greenwald limit = {stab['Greenwald_density_limit_1e20']:.2f} e20 m-3")
print(f"    betaN = {stab['betaN']:.4f}  Troyon margin = {stab['Troyon_margin']:.3f}")
print(f"    li = {stab['li']:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
psi2d = eq.psi()
sep   = eq.separatrix(npoints=360)
mag   = eq.magneticAxis()

# External ψ via Green volume integral (free-space, physical decay)
from gspack.boundary import greens_volume_psi
idx_i, idx_j = np.where(eq.plasma_mask)
R_src, Z_src = eq.R[idx_i, idx_j], eq.Z[idx_i, idx_j]
J_src = eq._Jtor[idx_i, idx_j]

R_ext = np.linspace(eq.Rmin - 0.3, eq.Rmax + 0.3, 97)
Z_ext = np.linspace(eq.Zmin - 0.3, eq.Zmax + 0.3, 97)
R_ext2d, Z_ext2d = np.meshgrid(R_ext, Z_ext, indexing='ij')
psi_ext = greens_volume_psi(
    R_ext2d.ravel(), Z_ext2d.ravel(),
    R_src, Z_src, J_src,
    eq.dR, eq.dZ).reshape(R_ext2d.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle(
    f"gspack v2.0 — Fixed-boundary  "
    f"R₀={R0}  a={a}  κ={KAPPA}  δ={DELTA}  "
    f"Ip={Ip_target:.0e}  βp={betap_target}",
    fontsize=12)

# Panel 1: ψ on large grid — FDM inside + Green outside
ax = axes[0]
cf = ax.contourf(R_ext2d, Z_ext2d, psi_ext, levels=30, cmap="RdYlBu_r")
# Internal contours (FDM, strict ψ=ψ_bndry on D-shape)
cs_in = ax.contour(eq.R, eq.Z, psi2d,
                    levels=np.linspace(eq.psi_bndry, eq.psi_axis, 12),
                    colors="white", linewidths=1.2)
ax.clabel(cs_in, inline=1, fontsize=7, fmt='%.3f')
# External contours (Green integral, free-space)
lev_ext = np.linspace(psi_ext.min() + 1e-6, eq.psi_bndry, 12)
cs_out = ax.contour(R_ext2d, Z_ext2d, psi_ext,
                     levels=lev_ext,
                     colors='cyan', linewidths=0.8, linestyles='--', alpha=0.7)
ax.clabel(cs_out, inline=1, fontsize=7, fmt='%.3f', colors='cyan')
ax.plot(sep[:, 0], sep[:, 1], "lime", lw=2, label="D-shape LCFS")
ax.plot(mag[0], mag[1], "k+", ms=12, mew=2, label="O-point")
ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_title("ψ — FDM interior (white) + Green exterior (cyan)")
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
plot_path = os.path.join(script_dir, "02_fixedboundary.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved  →  {os.path.basename(plot_path)}")
print("\n" + "=" * 64)
print("  Done.")
print("=" * 64)
