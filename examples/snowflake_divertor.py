"""
gspack v2.0 — 雪花偏滤器 (Snowflake Divertor) 示例
====================================================

雪花偏滤器是一种先进偏滤器配置，其特征是：
  • X 点处的极向磁场为 **二阶零点**（标准 X 点为一阶零点）
  • 即在 X 点处同时满足：
      Br = 0,   Bz = 0          ← 一阶零点（普通 X 点）
      ∂Br/∂Z = 0,  ∂Bz/∂R = 0  ← 二阶零点（雪花特征）
  • 这导致等离子体 "footprint" 分布在 6 条腿而不是 4 条，
    有效降低偏滤器靶板热通量

本示例演示：
  1. 标准双零偏滤器（DND）基准
  2. 雪花偏滤器（SF）——用 constrain_snowflake 实现二阶零点约束
  3. 对比两种构型的极向磁通分布和 X 点附近拓扑
  4. 量化雪花质量（二阶导数残差）

参考：Ryutov (2007) Phys. Plasmas 14, 064502
      Jeon (2015) JKPS 67(5) 843 — Section V
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RectBivariateSpline

from gspack.machine     import Coil, ShapedCoil, Machine, Wall, TestTokamak
from gspack.equilibrium import Equilibrium
from gspack.profiles    import ConstrainPaxisIp
from gspack.control     import constrain, constrain_snowflake
from gspack             import picard


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 机器定义 — 6 线圈雪花托卡马克
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def SnowflakeMachine():
    """
    6 线圈托卡马克，专为雪花偏滤器设计。

    线圈布局：
      P1L/P1U — 主极向场线圈（外侧上下对称）
      P2L/P2U — 外偏滤器线圈（标准位置）
      P3L/P3U — 内偏滤器线圈（靠近 X 点，雪花控制关键）

    线圈约束矩阵在 X 点处满秩（rank=6），
    可以独立控制零到二阶的所有分量。
    """
    coils = [
        ("P1L", ShapedCoil([(0.90,-1.20),(0.90,-1.10),(1.00,-1.10),(1.00,-1.20)])),
        ("P1U", ShapedCoil([(0.90, 1.20),(0.90, 1.10),(1.00, 1.10),(1.00, 1.20)])),
        ("P2L", Coil(1.75, -0.60)),
        ("P2U", Coil(1.75,  0.60)),
        ("P3L", Coil(0.85, -0.85)),   # 内偏滤器（下）
        ("P3U", Coil(0.85,  0.85)),   # 内偏滤器（上）
    ]
    wall = Wall(
        R=[0.70, 0.70, 1.55, 1.85, 1.85, 1.55],
        Z=[-0.95, 0.95, 0.95, 0.28, -0.28, -0.95],
    )
    return Machine(coils, wall)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 公共参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NX, NY   = 65, 65
P_AXIS   = 1e3           # 磁轴处压力 [Pa]
IP       = 2e5           # 等离子体电流 [A]
FVAC     = 1.0           # 真空 R·Bφ [T·m]
GAMMA    = 1e-12         # Tikhonov 正则化参数
MAXITS   = 50
RTOL     = 1e-3

# X 点目标位置（下/上）
XPT_L    = (1.00, -0.60)
XPT_U    = (1.00,  0.60)
OUTBOARD = (1.65,  0.00)  # 外中平面（等磁通约束点）

script_dir = os.path.dirname(os.path.abspath(__file__))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 辅助函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def snowflake_quality(eq, xpt, eps=5e-3):
    """
    计算 X 点处的雪花质量指标。

    返回 dict:
      Br, Bz          — 一阶零点残差（普通 X 点条件）
      dBr_dZ, dBz_dR  — 二阶零点残差（雪花条件）
      SF_index        — 雪花指数 = (|dBr/dZ| + |dBz/dR|) / (|Br| + |Bz| + 1e-30)
                         值越小，雪花质量越高
    """
    Rxp, Zxp = xpt
    Br = float(eq.Br(Rxp, Zxp))
    Bz = float(eq.Bz(Rxp, Zxp))
    dBr_dZ = float((eq.Br(Rxp, Zxp+eps) - eq.Br(Rxp, Zxp-eps)) / (2*eps))
    dBz_dR = float((eq.Bz(Rxp+eps, Zxp) - eq.Bz(Rxp-eps, Zxp)) / (2*eps))
    SF_index = (abs(dBr_dZ) + abs(dBz_dR)) / (abs(Br) + abs(Bz) + 1e-30)
    return {
        "Br": Br, "Bz": Bz,
        "dBr_dZ": dBr_dZ, "dBz_dR": dBz_dR,
        "SF_index": SF_index,
    }


def plot_equilibrium(ax, eq, title, xpt_markers=None, zoom_box=None,
                     show_sf_region=False):
    """在 ax 上绘制平衡磁通面。"""
    psi2d = eq.psi()
    R2d, Z2d = eq.R, eq.Z

    # 归一化磁通
    psiN = eq.psiN()

    # 磁通面（LCFS 以内）
    levels_in  = np.linspace(0, 1, 12)
    # Sort levels so matplotlib contour() gets them in increasing order
    levels_psi = np.sort(eq.psi_axis + levels_in * (eq.psi_bndry - eq.psi_axis))

    cf = ax.contourf(R2d, Z2d, psi2d, levels=30, cmap="RdYlBu_r", alpha=0.85)
    ax.contour(R2d, Z2d, psi2d, levels=levels_psi,
               colors="white", linewidths=0.5, alpha=0.7)
    # LCFS
    ax.contour(R2d, Z2d, psi2d, levels=[eq.psi_bndry],
               colors=["lime"], linewidths=2.2, zorder=5)

    # SOL 几条磁通面
    psi_sol = eq.psi_bndry + np.array([0.005, 0.012, 0.025]) * abs(eq.psi_axis - eq.psi_bndry)
    ax.contour(R2d, Z2d, psi2d, levels=psi_sol,
               colors=["orange"], linewidths=0.8, linestyles="--", alpha=0.6)

    # 磁轴
    if eq._opoints:
        ax.plot(eq._opoints[0][0], eq._opoints[0][1],
                "k+", ms=12, mew=2, zorder=6, label="O-point")

    # X 点标记
    if xpt_markers:
        for (Rx, Zx), style in xpt_markers:
            ax.plot(Rx, Zx, style, ms=10, mew=2, zorder=6)

    # 线圈
    for _, coil in eq.tokamak.coils:
        col = "red" if coil.current < 0 else "blue"
        ax.add_patch(plt.Circle((coil.R, coil.Z), 0.03,
                                 color=col, zorder=7, alpha=0.8))

    # 壁
    if eq.tokamak.wall is not None:
        w = eq.tokamak.wall
        ax.plot(w.R + [w.R[0]], w.Z + [w.Z[0]],
                "k-", lw=1.5, alpha=0.6)

    # 缩放框（X 点附近放大区域）
    if zoom_box:
        Rz, Zz, dRz, dZz = zoom_box
        rect = plt.Rectangle((Rz - dRz, Zz - dZz), 2*dRz, 2*dZz,
                              fill=False, edgecolor="yellow", lw=2, ls="--")
        ax.add_patch(rect)

    ax.set_xlabel("R [m]", fontsize=10)
    ax.set_ylabel("Z [m]", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(0.4, 2.0)
    ax.set_ylim(-1.0, 1.0)
    return cf


def plot_xpoint_zoom(ax, eq, xpt, title, levels=40, box_half=0.25):
    """放大显示 X 点附近的 |∇ψ| 分布。"""
    Rxp, Zxp = xpt
    R2d, Z2d = eq.R, eq.Z
    psi2d     = eq.psi()

    # |∇ψ|² 计算
    f = RectBivariateSpline(R2d[:,0], Z2d[0,:], psi2d)
    dpsidR = f(R2d[:,0], Z2d[0,:], dx=1)
    dpsidZ = f(R2d[:,0], Z2d[0,:], dy=1)
    grad_psi = np.sqrt(dpsidR**2 + dpsidZ**2)

    cf = ax.contourf(R2d, Z2d, grad_psi, levels=levels, cmap="plasma")
    ax.contour(R2d, Z2d, psi2d, levels=[eq.psi_bndry],
               colors=["lime"], linewidths=1.5)

    # 额外 SOL 磁通面
    psi_sol = eq.psi_bndry + np.array([0.003, 0.008, 0.016]) * abs(eq.psi_axis - eq.psi_bndry)
    ax.contour(R2d, Z2d, psi2d, levels=psi_sol,
               colors=["white"], linewidths=0.7, linestyles="--", alpha=0.7)

    ax.plot(Rxp, Zxp, "r*", ms=14, zorder=6, label=f"X-point ({Rxp},{Zxp})")

    ax.set_xlabel("R [m]", fontsize=9)
    ax.set_ylabel("Z [m]", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(Rxp - box_half, Rxp + box_half)
    ax.set_ylim(Zxp - box_half, Zxp + box_half)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    return cf


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 求解 1：标准双零偏滤器（基准）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 64)
print("  gspack v2.0 — 雪花偏滤器示例")
print("=" * 64)

print("\n━━━ [1/2] 标准双零偏滤器（DND）基准 ━━━")

tok_std  = SnowflakeMachine()
eq_std   = Equilibrium(tok_std, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                       nx=NX, ny=NY)
pro_std  = ConstrainPaxisIp(p_axis=P_AXIS, Ip=IP, fvac=FVAC)
con_std  = constrain(
    xpoints = [XPT_L, XPT_U],
    isoflux = [(XPT_L[0], XPT_L[1], XPT_U[0], XPT_U[1]),
               (XPT_L[0], XPT_L[1], OUTBOARD[0], OUTBOARD[1]),
               (XPT_U[0], XPT_U[1], OUTBOARD[0], OUTBOARD[1])],
    gamma   = GAMMA,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    errs_std = picard.solve(eq_std, pro_std, con_std,
                            maxits=MAXITS, rtol=RTOL,
                            anderson_m=5, convergenceInfo=True)

q_std = snowflake_quality(eq_std, XPT_L)
print(f"\n  标准 DND 收敛于第 {len(errs_std[0])} 步")
print(f"  Ip         = {eq_std.plasmaCurrent():.3e} A")
print(f"  ψ_axis     = {eq_std.psi_axis:.5f} Wb/rad")
print(f"  ψ_bndry    = {eq_std.psi_bndry:.5f} Wb/rad")
print(f"\n  X 点 ({XPT_L[0]}, {XPT_L[1]}) 处的雪花质量：")
print(f"    Br       = {q_std['Br']:+.4e}  (→ 0 为 X 点)")
print(f"    Bz       = {q_std['Bz']:+.4e}  (→ 0 为 X 点)")
print(f"    ∂Br/∂Z   = {q_std['dBr_dZ']:+.4e}  (→ 0 为雪花条件)")
print(f"    ∂Bz/∂R   = {q_std['dBz_dR']:+.4e}  (→ 0 为雪花条件)")
print(f"    SF指数    = {q_std['SF_index']:.3f}  (0 = 完美雪花)")

print("\n  线圈电流（DND）：")
for name, coil in tok_std.coils:
    print(f"    {name:4s}: {coil.current:+.0f} A")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 求解 2：雪花偏滤器（SF）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n━━━ [2/2] 雪花偏滤器（SF） ━━━")
print("  约束：X 点处 Br=Bz=0  AND  ∂Br/∂Z=∂Bz/∂R=0")

tok_sf  = SnowflakeMachine()
eq_sf   = Equilibrium(tok_sf, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                      nx=NX, ny=NY)
pro_sf  = ConstrainPaxisIp(p_axis=P_AXIS, Ip=IP, fvac=FVAC)
con_sf  = constrain_snowflake(
    xpoints    = [XPT_L, XPT_U],
    isoflux    = [(XPT_L[0], XPT_L[1], XPT_U[0], XPT_U[1]),
                  (XPT_L[0], XPT_L[1], OUTBOARD[0], OUTBOARD[1]),
                  (XPT_U[0], XPT_U[1], OUTBOARD[0], OUTBOARD[1])],
    sf_xpoints = [XPT_L, XPT_U],    # 上下两个 X 点都要求雪花条件
    gamma      = GAMMA,
    eps        = 5e-3,               # 二阶导数数值微分步长
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    errs_sf = picard.solve(eq_sf, pro_sf, con_sf,
                           maxits=MAXITS, rtol=RTOL,
                           anderson_m=5, convergenceInfo=True)

q_sf = snowflake_quality(eq_sf, XPT_L)
print(f"\n  雪花 SF 收敛于第 {len(errs_sf[0])} 步")
print(f"  Ip         = {eq_sf.plasmaCurrent():.3e} A")
print(f"  ψ_axis     = {eq_sf.psi_axis:.5f} Wb/rad")
print(f"  ψ_bndry    = {eq_sf.psi_bndry:.5f} Wb/rad")
print(f"\n  X 点 ({XPT_L[0]}, {XPT_L[1]}) 处的雪花质量：")
print(f"    Br       = {q_sf['Br']:+.4e}  (→ 0 为 X 点)")
print(f"    Bz       = {q_sf['Bz']:+.4e}  (→ 0 为 X 点)")
print(f"    ∂Br/∂Z   = {q_sf['dBr_dZ']:+.4e}  (→ 0 为雪花条件)")
print(f"    ∂Bz/∂R   = {q_sf['dBz_dR']:+.4e}  (→ 0 为雪花条件)")
print(f"    SF指数    = {q_sf['SF_index']:.3f}  (0 = 完美雪花)")

improvement = (q_std['SF_index'] - q_sf['SF_index']) / (q_std['SF_index'] + 1e-30) * 100
print(f"\n  SF指数改善: {improvement:+.1f}%"
      f"  ({q_std['SF_index']:.3f} → {q_sf['SF_index']:.3f})")

print("\n  线圈电流（SF）：")
for name, coil in tok_sf.coils:
    # Compare with DND
    curr_dnd = dict(tok_std.coils)[name].current
    delta = coil.current - curr_dnd
    print(f"    {name:4s}: {coil.current:+.0f} A  (Δ = {delta:+.0f} A vs DND)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n  绘图中...")

fig = plt.figure(figsize=(18, 11))
fig.suptitle(    
    "Snowflake Divertor (SF) vs Standard Double-Null Divertor (DND)",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.38,
                        top=0.93, bottom=0.06, left=0.06, right=0.97)

# ── 行 0：全局平衡磁通面图 ───────────────────────────────────────────────
ax_dnd  = fig.add_subplot(gs[0, 0])
ax_sf   = fig.add_subplot(gs[0, 2])

xpt_markers_dnd = [((XPT_L[0], XPT_L[1]), "rx"),
                   ((XPT_U[0], XPT_U[1]), "rx")]
cf1 = plot_equilibrium(ax_dnd, eq_std,
                       f"Standard DND\nIp={eq_std.plasmaCurrent():.2e} A",
                       xpt_markers=xpt_markers_dnd,
                       zoom_box=(XPT_L[0], XPT_L[1], 0.28, 0.28))
cf2 = plot_equilibrium(ax_sf, eq_sf,
                       f"Snowflake SF\nIp={eq_sf.plasmaCurrent():.2e} A",
                       xpt_markers=xpt_markers_dnd,
                       zoom_box=(XPT_L[0], XPT_L[1], 0.28, 0.28))

# Add coil legend
ax_dnd.plot([], [], "o", color="blue",  ms=7, label="Positive current coil")
ax_dnd.plot([], [], "o", color="red",   ms=7, label="Negative current coil")
ax_dnd.plot([], [], "rx", ms=8, mew=2,  label="Target X-point")
ax_dnd.plot([], [], "-",  color="lime", lw=2, label="LCFS")
ax_dnd.plot([], [], "--", color="orange", lw=1, label="SOL")
ax_dnd.legend(fontsize=7, loc="upper right", framealpha=0.8)

# ── 行 1：X 点附近放大（|∇ψ| 图） ────────────────────────────────────────
ax_zoom_dnd = fig.add_subplot(gs[0, 1])
ax_zoom_sf  = fig.add_subplot(gs[0, 3])

cf3 = plot_xpoint_zoom(ax_zoom_dnd, eq_std, XPT_L,
                       f"DND X-point Region  |∇ψ|\n"
                       f"SF Index = {q_std['SF_index']:.3f}  "
                       f"(∂Br/∂Z={q_std['dBr_dZ']:.2e})",
                       box_half=0.28)
cf4 = plot_xpoint_zoom(ax_zoom_sf,  eq_sf,  XPT_L,
                       f"SF X-point Region  |∇ψ|\n"
                       f"SF Index = {q_sf['SF_index']:.3f}  "
                       f"(∂Br/∂Z={q_sf['dBr_dZ']:.2e})",
                       box_half=0.28)
plt.colorbar(cf3, ax=ax_zoom_dnd, label="|∇ψ| [Wb/m²]", fraction=0.03)
plt.colorbar(cf4, ax=ax_zoom_sf,  label="|∇ψ| [Wb/m²]", fraction=0.03)

# ── 行 2：收敛历史 + 线圈电流对比 + 雪花质量雷达图 ─────────────────────
ax_conv   = fig.add_subplot(gs[1, 0])
ax_coils  = fig.add_subplot(gs[1, 1])
ax_sfq    = fig.add_subplot(gs[1, 2])
ax_bpol   = fig.add_subplot(gs[1, 3])

# 收敛历史
ax_conv.semilogy(np.arange(1, len(errs_std[1])+1), errs_std[1],
                 "b-o", ms=4, lw=1.5, label="DND")
ax_conv.semilogy(np.arange(1, len(errs_sf[1])+1),  errs_sf[1],
                 "r-s", ms=4, lw=1.5, label="SF")
ax_conv.axhline(RTOL, color="gray", ls="--", lw=1, label=f"rtol={RTOL}")
ax_conv.set_xlabel("Iteration steps", fontsize=9)
ax_conv.set_ylabel("Relative residual |Δψ|/span(ψ)", fontsize=9)
ax_conv.set_title("Picard Convergence (Anderson m=5)", fontsize=10)
ax_conv.legend(fontsize=8)
ax_conv.grid(True, alpha=0.3)

# 线圈电流对比柱状图
coil_names = [n for n, _ in tok_std.coils]
I_dnd = np.array([dict(tok_std.coils)[n].current for n in coil_names]) / 1e3
I_sf  = np.array([dict(tok_sf.coils)[n].current  for n in coil_names]) / 1e3
x = np.arange(len(coil_names))
w = 0.35
bars1 = ax_coils.bar(x - w/2, I_dnd, w, label="DND", color="steelblue", alpha=0.8)
bars2 = ax_coils.bar(x + w/2, I_sf,  w, label="SF",  color="tomato",    alpha=0.8)
ax_coils.axhline(0, color="black", lw=0.8)
ax_coils.set_xticks(x)
ax_coils.set_xticklabels(coil_names, fontsize=8)
ax_coils.set_ylabel("Coil current [kA]", fontsize=9)
ax_coils.set_title("PF Coil Current Comparison", fontsize=10)
ax_coils.legend(fontsize=8)
ax_coils.grid(True, alpha=0.3, axis="y")

# 雪花质量对比（4个指标的对数绝对值）
metrics = ["Br", "Bz", "|∂Br/∂Z|", "|∂Bz/∂R|"]
vals_dnd = [abs(q_std["Br"]), abs(q_std["Bz"]),
            abs(q_std["dBr_dZ"]), abs(q_std["dBz_dR"])]
vals_sf  = [abs(q_sf["Br"]),  abs(q_sf["Bz"]),
            abs(q_sf["dBr_dZ"]), abs(q_sf["dBz_dR"])]
x2 = np.arange(len(metrics))
ax_sfq.bar(x2 - 0.2, vals_dnd, 0.38, label="DND", color="steelblue", alpha=0.8)
ax_sfq.bar(x2 + 0.2, vals_sf,  0.38, label="SF",  color="tomato",    alpha=0.8)
ax_sfq.set_yscale("log")
ax_sfq.set_xticks(x2)
ax_sfq.set_xticklabels(metrics, fontsize=8)
ax_sfq.set_ylabel("Residual Magnitude [T or T/m]", fontsize=9)
ax_sfq.set_title("X-point Constraint Satisfaction\n(lower is better)", fontsize=10)
ax_sfq.legend(fontsize=8)
ax_sfq.grid(True, alpha=0.3, axis="y")
# Mark the target = 0 line
ax_sfq.axhline(1e-5, color="gray", ls=":", lw=1, label="Target ≈ 0")

# Poloidal field Bpol distribution along separatrix (proxy for "footprint")
theta_arr = np.linspace(0, 2*np.pi, 200, endpoint=False)
R_ax_std = eq_std._opoints[0][0]; Z_ax_std = eq_std._opoints[0][1]
R_ax_sf  = eq_sf._opoints[0][0];  Z_ax_sf  = eq_sf._opoints[0][1]

def sep_bpol(eq, R_ax, Z_ax, ntheta=200):
    """Bpol on the separatrix, parameterised by poloidal angle."""
    sep = eq.separatrix(npoints=ntheta)
    Bp  = eq.Bpol(sep[:, 0], sep[:, 1])
    # geometric poloidal angle from O-point
    theta = np.arctan2(sep[:, 1] - Z_ax, sep[:, 0] - R_ax)
    idx   = np.argsort(theta)
    return theta[idx], Bp[idx]

theta_dnd, Bp_dnd = sep_bpol(eq_std, R_ax_std, Z_ax_std)
theta_sf,  Bp_sf  = sep_bpol(eq_sf,  R_ax_sf,  Z_ax_sf)

ax_bpol.plot(np.degrees(theta_dnd), Bp_dnd * 1e3, "b-",  lw=1.5, label="DND")
ax_bpol.plot(np.degrees(theta_sf),  Bp_sf  * 1e3, "r--", lw=1.5, label="SF")
ax_bpol.set_xlabel("Poloidal angle θ [°]", fontsize=9)
ax_bpol.set_ylabel("Bpol [mT]", fontsize=9)
ax_bpol.set_title("Poloidal Field on Separatrix\n(SF reduces Bpol near X-point)", fontsize=10)
ax_bpol.legend(fontsize=8)
ax_bpol.grid(True, alpha=0.3)
# Mark X-point angle
for eq_i, col, lab in [(eq_std, "blue", "DND Xpt"),
                        (eq_sf,  "red",  "SF Xpt")]:
    if eq_i._xpoints:
        Rx, Zx = eq_i._xpoints[0][:2]
        R_a = eq_i._opoints[0][0]; Z_a = eq_i._opoints[0][1]
        theta_xp = np.degrees(np.arctan2(Zx - Z_a, Rx - R_a))
        ax_bpol.axvline(theta_xp, color=col, ls=":", lw=1.5, alpha=0.7)

plt.savefig(os.path.join(script_dir,  "snowflake_divertor.png"),
            dpi=130, bbox_inches="tight")
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 最终总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 64)
print("  最终对比总结")
print("=" * 64)
print(f"\n  {'指标':<22}  {'DND':>12}  {'SF':>12}  {'改善':>10}")
print("  " + "-" * 60)
for key, label, fmt in [
    ("Br",      "Br at Xpt [T]",     ".2e"),
    ("Bz",      "Bz at Xpt [T]",     ".2e"),
    ("dBr_dZ",  "∂Br/∂Z [T/m]",     ".2e"),
    ("dBz_dR",  "∂Bz/∂R [T/m]",     ".2e"),
    ("SF_index","SF 指数 (↓好)",     ".4f"),
]:
    vd = q_std[key]; vs = q_sf[key]
    impr = (abs(vd) - abs(vs)) / (abs(vd) + 1e-30) * 100
    print(f"  {label:<22}  {vd:>12{fmt}}  {vs:>12{fmt}}  {impr:>+9.1f}%")

print()
print("  物理解释：")
print("  • SF 约束迫使 ∂Br/∂Z 和 ∂Bz/∂R → 0 at X-point")
print("  • 这使得 X 点处的磁场拓扑从单 X 点变为 '六腿' 雪花形")
print("  • X 点附近 Bpol 更小，磁通面张开，分散偏滤器热通量")
print("  • 对应物理：flux expansion ↑，connection length ↑，peak heat load ↓")
print()
print("  图像已保存 → snowflake_divertor.png")
print("=" * 64)
