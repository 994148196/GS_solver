"""
examples/01_freeboundary.py
============================
Reproduce FreeGS 01-freeboundary.py using gspack (no FreeGS dependency).

Prints the same quantitative outputs as the FreeGS reference:
  - Plasma current, psi_axis, psi_bndry
  - Magnetic axis, Shafranov shift
  - Shape: a, kappa, delta, R/a
  - Betas: bp, bt, bN
  - li, Volume
  - q profile
  - Coil currents
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gspack.machine     import TestTokamak
from gspack.equilibrium import Equilibrium
from gspack.profiles    import ConstrainPaxisIp
from gspack.control     import constrain
from gspack             import picard

# ─────────────────────────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────────────────────────

print("=" * 64)
print("  gspack  —  01-freeboundary  (FreeGS reproduction)")
print("=" * 64)

tok = TestTokamak()
print("\n  Machine: TestTokamak")
for name, coil in tok.coils:
    print(f"    {name:5s}  {coil}")

eq = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65)
print(f"\n  Grid: {eq.nx}×{eq.ny}")

pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)

con = constrain(
    xpoints=[(1.1, -0.6), (1.1, 0.6)],
    isoflux=[(1.1,-0.6, 1.1, 0.6),
             (1.1,-0.6, 1.7, 0.0),
             (1.1, 0.6, 1.7, 0.0)],
    gamma=1e-12,
)

# ─────────────────────────────────────────────────────────────────
#  Solve
# ─────────────────────────────────────────────────────────────────

print("\n  Picard iteration:\n")
picard.solve(eq, pro, con, maxits=50, rtol=1e-3, convergenceInfo=True)

# ─────────────────────────────────────────────────────────────────
#  Diagnostics
# ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  RESULTS")
print("=" * 64)

Ip_calc = eq.plasmaCurrent()
print(f"\n  Plasma current Ip  = {Ip_calc:.6e} A")
print(f"  psi axis           = {eq.psi_axis:.6f} Wb/rad")
print(f"  psi boundary       = {eq.psi_bndry:.6f} Wb/rad")

mag_ax = eq.magneticAxis()
print(f"\n  Magnetic axis: R = {mag_ax[0]:.4f} m,  Z = {mag_ax[1]:.2e} m")

geo_ax = eq.geometricAxis()
shaf   = eq.shafranovShift()
print(f"  Geometric axis: R = {geo_ax[0]:.4f} m")
print(f"  Shafranov shift dR = {shaf[0]*100:.3f} cm")

a     = eq.minorRadius()
kappa = eq.elongation()
delta = eq.triangularity()
Ara   = eq.aspectRatio()
print(f"\n  Minor radius a  = {a:.4f} m")
print(f"  Elongation kappa= {kappa:.4f}")
print(f"  Triangularity d = {delta:.4f}")
print(f"  Aspect ratio R/a= {Ara:.4f}")

bp = eq.poloidalBeta()
bt = eq.toroidalBeta()
bn = eq.betaN()
li = eq.internalInductance()
V  = eq.plasmaVolume()
print(f"\n  Poloidal beta bp = {bp:.6f}")
print(f"  Toroidal beta bt = {bt:.6f}")
print(f"  Normalised bN    = {bn:.4f}")
print(f"  Int. inductance  = {li:.4f}")
print(f"  Plasma volume V  = {V:.4f} m^3")

psiN_q = np.array([0.05, 0.25, 0.50, 0.75, 0.95])
q_vals = eq.q(psiN_q)
print(f"\n  Safety factor q(psiN):")
for pn, qv in zip(psiN_q, q_vals):
    print(f"    q({pn:.2f}) = {qv:.4f}")

Ri, Ro = eq.innerOuterSeparatrix()
print(f"\n  Separatrix:  Rinner={Ri:.4f} m,  Router={Ro:.4f} m")

print("\n  PF Coil currents:")
tok.printCurrents()

# ─────────────────────────────────────────────────────────────────
#  Plot
# ─────────────────────────────────────────────────────────────────

R2D  = eq.R
Z2D  = eq.Z
psi  = eq.psi()
psiN = eq.psiN()
sep  = eq.separatrix()

psiN_full = np.linspace(0.02, 0.98, 60)
q_full    = eq.q(psiN_full)
psiN_1d   = np.linspace(0.0, 1.0, 100)
p_1d      = np.array([eq.pressure(p) for p in psiN_1d])

fig = plt.figure(figsize=(16, 11))
gs_lay = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.38)

def draw_machine(ax):
    if tok.wall:
        Rw = tok.wall.R + [tok.wall.R[0]]
        Zw = tok.wall.Z + [tok.wall.Z[0]]
        ax.plot(Rw, Zw, "k--", lw=1.2, alpha=0.5)
    for name, coil in tok.coils:
        ax.plot(coil.R, coil.Z, "bs", ms=6)

# A: psi
axA = fig.add_subplot(gs_lay[:2,0]); axA.set_aspect("equal")
cf = axA.contourf(R2D, Z2D, psi, levels=60, cmap="RdYlBu_r")
plt.colorbar(cf, ax=axA, label="psi [Wb/rad]", fraction=0.046)
lvls = np.linspace(min(eq.psi_axis, eq.psi_bndry), max(eq.psi_axis, eq.psi_bndry), 20)
axA.contour(R2D, Z2D, psi, levels=lvls, colors="k", linewidths=0.6, alpha=0.6)
axA.contour(R2D, Z2D, psi, levels=[eq.psi_bndry], colors="lime", linewidths=2.5)
draw_machine(axA)
axA.plot(mag_ax[0], mag_ax[1], "w+", ms=12, mew=2.5)
for Rx, Zx in [(1.1,-0.6),(1.1,0.6)]: axA.plot(Rx, Zx, "rx", ms=10, mew=2.5)
axA.set_xlabel("R [m]"); axA.set_ylabel("Z [m]"); axA.set_title("Poloidal Flux psi")

# B: psiN + separatrix
axB = fig.add_subplot(gs_lay[:2,1]); axB.set_aspect("equal")
cf2 = axB.contourf(R2D, Z2D, psiN, levels=60, cmap="plasma_r", vmin=0, vmax=1.05)
plt.colorbar(cf2, ax=axB, label="psiN", fraction=0.046)
axB.plot(sep[:,0], sep[:,1], "lime", lw=2.5, label="Sep.")
draw_machine(axB)
axB.set_xlabel("R [m]"); axB.set_ylabel("Z [m]")
axB.set_title(f"Norm. Flux  kappa={kappa:.2f}  delta={delta:.2f}")
axB.legend(fontsize=7)

# C: Jtor
axC = fig.add_subplot(gs_lay[:2,2]); axC.set_aspect("equal")
Jmask = np.where(psiN<=1, eq.Jtor, np.nan)
vmax_J = np.nanpercentile(Jmask, 98)
cf3 = axC.contourf(R2D, Z2D, Jmask, levels=60, cmap="hot", vmin=0, vmax=vmax_J)
plt.colorbar(cf3, ax=axC, label="Jphi [A/m^2]", fraction=0.046)
axC.contour(R2D, Z2D, psi, levels=[eq.psi_bndry], colors="cyan", linewidths=2)
draw_machine(axC)
axC.set_xlabel("R [m]"); axC.set_ylabel("Z [m]"); axC.set_title("Toroidal Current Jphi")

# D: Bpol
axD = fig.add_subplot(gs_lay[:2,3]); axD.set_aspect("equal")
Bp2d = eq.Bpol()
cf4  = axD.contourf(R2D, Z2D, Bp2d, levels=60, cmap="viridis")
plt.colorbar(cf4, ax=axD, label="|Bpol| [T]", fraction=0.046)
axD.contour(R2D, Z2D, psi, levels=[eq.psi_bndry], colors="red", linewidths=2)
draw_machine(axD)
axD.set_xlabel("R [m]"); axD.set_ylabel("Z [m]"); axD.set_title("|Bpol|")

# E: pressure
axE = fig.add_subplot(gs_lay[2,0])
axE.plot(psiN_1d, p_1d/1e3, "b-", lw=2)
axE.fill_between(psiN_1d, 0, p_1d/1e3, alpha=0.15, color="blue")
axE.set_xlabel("psiN"); axE.set_ylabel("p [kPa]")
axE.set_title("Pressure p(psiN)"); axE.grid(True, alpha=0.3)

# F: q profile
axF = fig.add_subplot(gs_lay[2,1])
axF.plot(psiN_full, q_full, "r-", lw=2.5)
for qref in [1,2,3]:
    axF.axhline(qref, color="gray", ls="--", lw=1)
axF.set_xlabel("psiN"); axF.set_ylabel("q")
axF.set_title(f"Safety Factor q  (q95={q_vals[-1]:.2f})")
axF.set_xlim(0,1); axF.set_ylim(0, min(q_full.max()*1.2, 10))
axF.grid(True, alpha=0.3)

# G: coil currents
axG = fig.add_subplot(gs_lay[2,2])
names  = [n for n,_ in tok.coils]
amps   = [c.current/1e3 for _,c in tok.coils]
colors = ["steelblue" if v>=0 else "tomato" for v in amps]
bars = axG.bar(names, amps, color=colors, edgecolor="k", width=0.6)
for bar, v in zip(bars, amps):
    axG.text(bar.get_x()+bar.get_width()/2, bar.get_height()+(2 if v>=0 else -2),
             f"{v:.1f}", ha="center", va="bottom" if v>=0 else "top", fontsize=8, fontweight="bold")
axG.axhline(0, color="k", lw=0.8); axG.set_ylabel("Current [kA]")
axG.set_title("PF Coil Currents"); axG.grid(True, alpha=0.3, axis="y")

# H: summary
axH = fig.add_subplot(gs_lay[2,3]); axH.axis("off")
txt = (
    f"gspack v1.0\nTestTokamak  65×65\n{'─'*24}\n"
    f"Ip     = {Ip_calc/1e3:7.2f} kA\n"
    f"psi_ax = {eq.psi_axis:7.5f}\n"
    f"psi_bn = {eq.psi_bndry:7.5f}\n{'─'*24}\n"
    f"R_mag  = {mag_ax[0]:7.4f} m\n"
    f"dR_Sh  = {shaf[0]*100:7.3f} cm\n"
    f"a      = {a:7.4f} m\n"
    f"kappa  = {kappa:7.4f}\n"
    f"delta  = {delta:7.4f}\n"
    f"R/a    = {Ara:7.4f}\n{'─'*24}\n"
    f"bp     = {bp:7.6f}\n"
    f"bN     = {bn:7.4f}\n"
    f"li     = {li:7.4f}\n"
    f"V      = {V:7.4f} m^3\n{'─'*24}\n"
    f"q(0.50)= {q_vals[2]:7.4f}\n"
    f"q(0.95)= {q_vals[4]:7.4f}\n"
)
axH.text(0.02, 0.98, txt, transform=axH.transAxes, fontsize=7.5, va="top",
         fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

fig.suptitle(
    f"gspack  01-freeboundary  |  Ip={Ip_calc/1e3:.0f} kA  "
    f"kappa={kappa:.2f}  delta={delta:.2f}  R/a={Ara:.2f}",
    fontsize=12, fontweight="bold", y=1.01)

outpath = os.path.join(os.path.dirname(__file__), "..", "01_freeboundary_gspack.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved  ->  01_freeboundary_gspack.png")
print("\n" + "=" * 64 + "\n  Done.\n" + "=" * 64)
