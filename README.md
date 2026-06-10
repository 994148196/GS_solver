# gspack v2.0 — Grad-Shafranov 托卡马克平衡求解器

> Grad-Shafranov Plasma Equilibrium Code for Tokamak — 支持自由边界和固定边界两种模式

---

## 目录

- [物理背景](#物理背景)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [自由边界求解](#自由边界求解)
- [固定边界求解](#固定边界求解)
- [GPU 加速](#gpu-加速)
- [示例列表](#示例列表)
- [项目结构](#项目结构)
- [参考文献](#参考文献)

---

## 物理背景

`gspack` 求解轴对称托卡马克中的 Grad-Shafranov 方程：

$$\Delta^*\psi \equiv R\frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right) + \frac{\partial^2\psi}{\partial Z^2} = -\mu_0 R J_\phi(R, Z)$$

其中 $\psi$ 是极向磁通函数，$J_\phi$ 是环向电流密度。方程等价形式：

$$\Delta^*\psi = -\mu_0 R^2 p'(\psi) - f f'(\psi)$$

求解器支持两种边界模式：

| 模式 | 说明 | 边界条件 |
|------|------|----------|
| **自由边界** | 等离子体被线圈约束，边界由 X 点/限幅器确定 | 矩形域上 ψ 通过 von Hagenow 方法 + 线圈 Green 函数确定 |
| **固定边界** | 预设 D 形 LCFS（给定 $R_0, a, \kappa, \delta$），无线圈 | D 形轮廓上 $\psi = \psi_{\text{bndry}}$，自洽由 Green 体积分确定 |

电流剖面采用 **Jeon (2015)** 参数化形式：

$$J_\phi = L \left[\beta_0 \frac{R}{R_0} + (1-\beta_0)\frac{R_0}{R}\right]
(1 - \hat{\psi}^{\alpha_m})^{\alpha_n}$$

两个线性参数 $L, \beta_0$ 由全局约束（$I_p, \beta_p$ 或 $p_0, I_p$）确定。

---

## 功能特性

### 核心求解

| 功能 | 模块 | 说明 |
|------|------|------|
| 二阶/四阶 FDM | `greens.py` | `order=2` 或 `order=4`，四阶精度提高 10 倍 |
| LU / AMG 求解器 | `greens.py` | `method='auto'` 自动选择；AMG 支持 513×513 网格 |
| Anderson 加速 Picard | `picard.py` | 历史窗口 m=5，收敛加速 ~3 倍 |
| CPU / GPU 透明后端 | `backend.py` | 自动选择 CuPy 或 NumPy |
| 掩码求解器 | `greens.py` | 任意形状内部 GS、外部 Dirichlet BC |

### 自由边界

| 功能 | 模块 | 说明 |
|------|------|------|
| von Hagenow 方法 | `boundary.py` | 4 阶法向导数 + Green 边界积分 |
| 线圈电流控制 | `control.py` | X 点/等磁通约束，Tikhonov 正则化 |
| Snowflake 偏滤器 | `control.py` | 二阶 X 点零场约束 |
| 限幅器等离子体 | `equilibrium.py` | `check_limited=True` |
| 形状优化 | `optimize.py` | 优化线圈电流达目标 $\kappa, \delta, R$ |
| Monte Carlo UQ | `optimize.py` | 线圈电流/位置误差分析 |

### 固定边界

| 功能 | 模块 | 说明 |
|------|------|------|
| D 形 LCFS | `boundary.py` | Cerfon–Solovev 风格参数化 |
| 自洽 Green 边界条件 | `equilibrium.py` | 每迭代步从 Green 体积分确定 $\psi_{\text{bndry}}$ |
| 外部真空区求解 | `equilibrium.py` | `psi_on_grid()` 任意网格 Green 体积分 |
| 数据集生成 | `04_generate_dataset.py` | LHS 采样 8 参数 → $\psi(R,Z)$ HDF5 |

### 电流剖面

| 剖面类 | 约束条件 |
|--------|----------|
| `ConstrainPaxisIp` | 固定轴心压力 $p_0$ + 总电流 $I_p$ |
| `ConstrainBetapIp` | 固定极向比压 $\beta_p$ + 总电流 $I_p$ |
| `ConstrainRotation` | 固定 $p_0, I_p$ + 环向旋转 $\Omega(\psi)$ |
| `ProfilesPprimeFfprime` | 直接指定 $p'(\psi)$ 和 $ff'(\psi)$ |

### 诊断

| 功能 | 模块 | 说明 |
|------|------|------|
| O 点 / X 点检测 | `critical.py` | Newton 精化 + 单调性滤波 |
| LCFS 追踪 | `separatrix.py` | 二分射线追踪 |
| 安全因子 q | `safety.py` | 通量面追踪积分 |
| 通量面坐标 | `diagnostics.py` | $(\psi, \theta)$ 网格，供输运代码 |
| Poincaré 截面 | `diagnostics.py` | 磁力线追踪 |
| MHD 稳定性 | `diagnostics.py` | q 面、Greenwald、Troyon $\beta$ |

### I/O

| 功能 | 模块 | 说明 |
|------|------|------|
| G-EQDSK | `geqdsk.py` | 读写标准平衡文件，兼容 TRANSP/VMEC |
| HDF5 存档 | `io.py` | 完整平衡数据持久化 |

---

## 快速开始

### 安装依赖

```bash
# 核心
pip install numpy scipy matplotlib

# 可选
pip install h5py          # HDF5 I/O
pip install pyamg         # 大网格 AMG 求解器
pip install cupy-cuda12x  # GPU 加速
pip install tqdm          # 数据集生成进度条
```

### 运行示例

```bash
cd gspack2_TRAE

# 自由边界（TestTokamak + PF 线圈）
python examples/01_freeboundary.py

# 固定边界（D 形 LCFS，自洽 Green 边界条件）
python examples/02_fixedboundary.py

# 固定边界 + 外部真空区 Green 体积分
python examples/03_fixedboundary_largegrid.py

# 生成数据集（LHS 采样 → HDF5）
python examples/04_generate_dataset.py --nsamples 100

# 数据集可视化
python examples/05_visualize_dataset.py
```

### 运行测试

```bash
python -m pytest tests/ -v
```

---

## 自由边界求解

```python
from gspack.machine     import TestTokamak
from gspack.equilibrium import Equilibrium
from gspack.profiles    import ConstrainPaxisIp
from gspack.control     import constrain
from gspack             import picard

# 1. 机器（含 6 组 PF 线圈）
tok = TestTokamak()

# 2. 平衡对象
eq = Equilibrium(tok, nx=65, ny=65, order=2, method='auto')

# 3. 电流剖面（p₀ = 10 kPa, Ip = 200 kA）
pro = ConstrainPaxisIp(p_axis=1e4, Ip=2e5, fvac=1.0, Raxis=1.0)

# 4. X 点约束
con = constrain(
    xpoints=[(1.1, -0.6), (1.1, 0.6)],
    isoflux=[(1.1, -0.6, 1.1, 0.6),
             (1.1, -0.6, 1.7, 0.0),
             (1.1,  0.6, 1.7, 0.0)])

# 5. Picard 迭代
picard.solve(eq, pro, con, anderson_m=5)

# 6. 诊断
print(f"Ip    = {eq.plasmaCurrent():.2e} A")
print(f"βp    = {eq.poloidalBeta():.4f}")
print(f"ψ_axis= {eq.psi_axis:.6f} Wb/rad")
print(f"q(95) = {eq.q(np.array([0.95]))[0]:.3f}")
```

---

## 固定边界求解

```python
from gspack.equilibrium import FixedBoundaryEquilibrium
from gspack.profiles    import ConstrainBetapIp
from gspack             import picard

# 1. D 形 LCFS（R₀=1.0, a=0.5, κ=1.6, δ=0.33）
eq = FixedBoundaryEquilibrium(
    R0=1.0, a=0.5, kappa=1.6, delta=0.33,
    nx=65, ny=65)

# 2. 电流剖面（βp = 0.8, Ip = 200 kA）
pro = ConstrainBetapIp(
    betap=0.8, Ip=2e5, fvac=1.0,
    alpha_m=1.0, alpha_n=2.0, Raxis=1.0)

# 3. 求解（constrain=None 表示不使用线圈）
picard.solve(eq, pro, constrain=None, rtol=1e-3)

# 4. 结果
print(f"Ip    = {eq.plasmaCurrent():.2e} A  (target 2e5)")
print(f"βp    = {eq.poloidalBeta():.4f}  (target 0.8)")
print(f"ψ_a   = {eq.psi_axis:.4f} Wb/rad")
print(f"ψ_b   = {eq.psi_bndry:.4f} Wb/rad")

# 5. 外部真空区 ψ（任意网格）
R_v = np.linspace(0.1, 2.5, 97)
Z_v = np.linspace(-1.2, 1.2, 97)
R_v2d, Z_v2d = np.meshgrid(R_v, Z_v, indexing='ij')
psi_ext = eq.psi_on_grid(R_v2d, Z_v2d)
```

### 自洽边界条件说明

固定边界求解的核心是 **自洽迭代**：

1. 从当前 $\psi$ 计算 $J_\phi$（满足 $I_p, \beta_p$ 约束）
2. 在 D 形轮廓上计算 Green 体积分 $\psi_{\text{green}} = \iint G \cdot J_\phi \, dS$
3. $\psi_{\text{bndry}} = \langle \psi_{\text{green}} \rangle_{\text{LCFS}}$
4. 以 $\psi = \psi_{\text{bndry}}$ 为 Dirichlet BC 在 D 形内部求解 GS 方程
5. 迭代至收敛后，Green 体积分在 D 形上的值 $\approx \psi_{\text{bndry}}$

收敛后 `psi_on_grid()` 给出的自由空间 Green 体积分在 D 形上等于 $\psi_{\text{bndry}}$，
在外部真空区自动满足 Laplace 方程。

---

## GPU 加速

```python
import gspack.backend as bk

bk.set_backend('gpu')    # 需安装 cupy-cuda12x
bk.set_backend('cpu')    # 切回 CPU
bk.set_backend('auto')   # 自动检测

print(f"当前后端: {bk.get_backend()}")
```

Green 函数计算在 GPU 上可获得 2-5 倍加速（129×129 以上网格效果显著）。
稀疏矩阵求解始终在 CPU（SciPy 限制）。

---

## 示例列表

| 示例 | 文件 | 说明 |
|------|------|------|
| 01 | `01_freeboundary.py` | 自由边界 GS 平衡，TestTokamak + 6 PF 线圈 |
| 02 | `02_fixedboundary.py` | 固定边界平衡，D 形 LCFS + 自洽 Green BC |
| 03 | `03_fixedboundary_largegrid.py` | 固定边界 + 外部真空区 Green 体积分 |
| 04 | `04_generate_dataset.py` | LHS 采样 8 参数 → $\psi(R,Z)$ HDF5 数据集 |
| 05 | `05_visualize_dataset.py` | 数据集中随机抽样绘制 $\psi$ 图 |

---

## 项目结构

```
gspack2_TRAE/
├── gspack/                    # 核心库
│   ├── __init__.py            # 模块导出
│   ├── backend.py             # CPU/GPU 透明后端
│   ├── greens.py              # Green 函数 + GS 稀疏矩阵 + 求解器工厂
│   ├── equilibrium.py         # Equilibrium + FixedBoundaryEquilibrium
│   ├── boundary.py            # 自由/固定边界求解器 + D 形 LCFS 工具
│   ├── profiles.py            # 电流剖面（4 种约束模式）
│   ├── picard.py              # Anderson 加速 Picard 迭代
│   ├── critical.py            # O 点 / X 点检测
│   ├── separatrix.py          # LCFS 追踪
│   ├── safety.py              # 安全因子 q
│   ├── diagnostics.py         # Poincaré 截面、通量面坐标、MHD 稳定性
│   ├── machine.py             # 线圈 + 机器定义
│   ├── control.py             # 等离子体控制（X 点/等磁通约束）
│   ├── io.py                  # HDF5 存档
│   ├── geqdsk.py              # G-EQDSK 读写
│   └── optimize.py            # 形状优化 + Monte Carlo UQ
├── examples/                  # 示例
│   ├── 01_freeboundary.py
│   ├── 02_fixedboundary.py
│   ├── 03_fixedboundary_largegrid.py
│   ├── 04_generate_dataset.py
│   └── 05_visualize_dataset.py
├── tests/                     # 测试
│   ├── test_gspack.py
│   └── test_gspack_v2.py
├── docs/
│   └── gspack_documentation.md  # 详细文档（物理 + API + 示例）
├── refs/
│   └── Jeon_JKPS15_Development of a free boundary Tokamak ...
└── datasets/                  # 生成的数据集（自动创建）
```

---

## 参考文献

1. **Jeon, Y.M. (2015)**. "Development of a free boundary Tokamak Equilibrium Solver (TES) for Advanced Study of Tokamak Equilibria." *Journal of the Korean Physical Society*, 67(5), 843-853. [`refs/`](refs/)
2. **Cerfon, A.J. & Solovev, I.V. (2012)**. "One-size-fits-all analytic solutions to the Grad–Shafranov equation." *Physics of Plasmas*, 19, 032506.
3. **von Hagenow, K. (1969)**. "A method for the solution of the free-boundary magnetohydrodynamic equilibrium equation." IPP 6/106.
4. **Dudson, B. & contributors (2016)**. FreeGS: Free boundary Grad-Shafranov solver. https://github.com/freegs-plasma/freegs
5. **Shafranov, V.D. (1957)**. "On magnetohydrodynamical equilibrium configurations." *Sov. Phys. JETP*, 6, 545.
