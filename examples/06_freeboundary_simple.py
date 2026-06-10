"""
gspack v2.0 — 06-freeboundary-simple
=====================================
Simplified free-boundary solver using constrain_axis.

Unlike 01-freeboundary which requires specifying X-point positions and
isoflux pairs (semi-fixed-boundary), this example demonstrates a simpler
approach: only constrain the magnetic axis position via Br=0, Bz=0.

The profile objects (ConstrainPaxisIp / ConstrainBetapIp) handle the
I_p (or β_p) constraint — together they provide complete free-boundary
control without needing divertor geometry specifications.

Key benefits over the original constrain(xpoints+, isoflux+):
  • Magnetic axis stays exactly at the target position (R₀, Z₀)
  • Faster convergence (11-12 iterations vs 15-26)
  • More stable — no need to tune X-point locations
  • Clean separation: profiles → plasma current; constrain_axis → position
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gspack.machine     import TestTokamak
from gspack.equilibrium import Equilibrium
from gspack.profiles    import ConstrainPaxisIp, ConstrainBetapIp
from gspack.control     import constrain_axis
from gspack             import picard, geqdsk, diagnostics
import gspack.backend as bk

# ── Config ────────────────────────────────────────────────────────────────────
NX, NY      = 65, 65
ORDER       = 2
METHOD      = "lu"
ANDERSON_M  = 5
BACKEND     = "cpu"

bk.set_backend(BACKEND)
script_dir = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
#  Example 1: ConstrainPaxisIp + constrain_axis
#  Fix p_axis=1e3 Pa, Ip=200 kA, and magnetic axis at R₀=1.0 m
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("  Example 1: constrain_axis + ConstrainPaxisIp")
print("  p_axis=1e3 Pa, Ip=200 kA, axis at R₀=1.0 m")
print("=" * 64)

tok1 = TestTokamak()
eq1  = Equilibrium(tok1, nx=NX, ny=NY, order=ORDER, method=METHOD)
pro1 = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
con1 = constrain_axis(R0=1.0, Z0=0.0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    errs1 = picard.solve(eq1, pro1, con1, maxits=50, rtol=1e-3,
                         anderson_m=ANDERSON_M, convergenceInfo=True)

mag1 = eq1.magneticAxis()
print(f"\n  Ip      = {eq1.plasmaCurrent():.4e} A  (target 2.00e5)")
print(f"  Axis    = ({mag1[0]:.4f}, {mag1[1]:.2e}) m  (target 1.0, 0.0)")
print(f"  psi_a   = {eq1.psi_axis:.6f}  psi_b = {eq1.psi_bndry:.6f}")
print(f"  βp      = {eq1.poloidalBeta():.4f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Example 2: ConstrainBetapIp + constrain_axis
#  Fix βp=0.8, Ip=200 kA, and magnetic axis at R₀=1.0 m
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("  Example 2: constrain_axis + ConstrainBetapIp")
print("  βp=0.8, Ip=200 kA, axis at R₀=1.0 m")
print("=" * 64)

tok2 = TestTokamak()
eq2  = Equilibrium(tok2, nx=NX, ny=NY, order=ORDER, method=METHOD)
# Use D-shaped initial psi (more stable for ConstrainBetapIp)
from gspack.boundary import dshape_lcfs, initial_psi_lcfs
Rl, Zl = dshape_lcfs(1.0, 0.5, 1.6, 0.33)
eq2.plasma_psi = initial_psi_lcfs(eq2.R, eq2.Z, Rl, Zl,
                                  psi_axis=1.0, psi_bndry=0.0)
eq2._update_boundary_psi()
pro2 = ConstrainBetapIp(betap=0.8, Ip=2e5, fvac=1.0,
                         alpha_m=1.0, alpha_n=2.0, Raxis=1.0)
con2 = constrain_axis(R0=1.0, Z0=0.0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    errs2 = picard.solve(eq2, pro2, con2, maxits=50, rtol=1e-3,
                         anderson_m=ANDERSON_M, convergenceInfo=True)

mag2 = eq2.magneticAxis()
print(f"\n  Ip      = {eq2.plasmaCurrent():.4e} A  (target 2.00e5)")
print(f"  βp      = {eq2.poloidalBeta():.4f}     (target 0.8)")
print(f"  Axis    = ({mag2[0]:.4f}, {mag2[1]:.2e}) m  (target 1.0, 0.0)")
print(f"  psi_a   = {eq2.psi_axis:.6f}  psi_b = {eq2.psi_bndry:.6f}")

# LCFS parameters
kappa = eq2.elongation(); delta = eq2.triangularity()
a     = eq2.minorRadius()
print(f"  LCFS    = R₀={mag2[0]:.3f} a={a:.3f} κ={kappa:.3f} δ={delta:.3f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  Plot comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (eq, title) in enumerate([
    (eq1, r"ConstrainPaxisIp ($p_0$=1kPa, $I_p$=200kA)"),
    (eq2, r"ConstrainBetapIp ($\beta_p$=0.8, $I_p$=200kA)"),
]):
    ax = axes[i]
    psi = eq.psi()
    levels = np.linspace(eq.psi_bndry, eq.psi_axis, 15)
    cf = ax.contourf(eq.R, eq.Z, psi, levels=levels, cmap="viridis")
    cs = ax.contour(eq.R, eq.Z, psi, levels=levels, colors="white",
                     linewidths=0.5, linestyles="dotted")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # LCFS (separatrix)
    sep = eq.separatrix()
    ax.plot(sep[:, 0], sep[:, 1], "w-", linewidth=2, label="LCFS")

    # Magnetic axis
    mag = eq.magneticAxis()
    ax.plot(mag[0], mag[1], "w*", markersize=12, markeredgecolor="k")

    # Coils
    for name, coil in tok1.coils:
        ax.plot(coil.R, coil.Z, "rs" if coil.current > 0 else "bs",
                markersize=6, fillstyle="none")
        ax.annotate(name, (coil.R + 0.05, coil.Z + 0.05), fontsize=7)

    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(script_dir, "06_freeboundary_simple.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"  Plot saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
#  G-EQDSK output (optional)
# ══════════════════════════════════════════════════════════════════════════════
geqdsk.write(eq2, os.path.join(script_dir, "06_freeboundary_simple.geqdsk"))
print(f"  EQDSK saved: 06_freeboundary_simple.geqdsk")

print("\n  Done.")
