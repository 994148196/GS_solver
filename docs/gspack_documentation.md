# gspack v2.0 — Documentation

`gspack` 是一个 **Grad-Shafranov 平衡求解器**（Grad-Shafranov Plasma Equilibrium
Code），用于托卡马克等离子体平衡的数值计算。支持 **自由边界**（Free-Boundary）
和 **固定边界**（Fixed-Boundary）两种模式。

**核心物理**：求解轴对称托卡马克中的 Grad-Shafranov 方程

$$\Delta^* \psi \equiv R\frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right) + \frac{\partial^2\psi}{\partial Z^2} = -\mu_0 R J_\phi(R, Z)$$

其中 $\psi$ 是极向磁通函数（poloidal magnetic flux），$J_\phi$ 是环向电流密度，
$\mu_0 = 4\pi \times 10^{-7}$ H/m 是真空磁导率。

---

## 目录

1. [物理模型](#1-物理模型)
   - [Grad-Shafranov 方程](#11-grad-shafranov-方程)
   - [电流剖面参数化（Jeon 2015）](#12-电流剖面参数化jeon-2015)
   - [Green 函数法](#13-green-函数法)
   - [von Hagenow 自由边界方法](#14-von-hagenow-自由边界方法)
   - [自洽固定边界方法](#15-自洽固定边界方法)
2. [安装与依赖](#2-安装与依赖)
3. [快速开始](#3-快速开始)
   - [自由边界求解](#31-自由边界求解)
   - [固定边界求解](#32-固定边界求解)
   - [外部区域求解（Green 体积分）](#33-外部区域求解green-体积分)
4. [API 参考](#4-api-参考)
   - [gspack.backend](#41-backend--cpu-gpu-后端)
   - [gspack.greens](#42-greens--green-函数与gs算子)
   - [gspack.equilibrium](#43-equilibrium--平衡类)
   - [gspack.boundary](#44-boundary--边界条件)
   - [gspack.profiles](#45-profiles--等离子体电流剖面)
   - [gspack.picard](#46-picard--picard-迭代)
   - [gspack.critical](#47-critical--o点和x点检测)
   - [gspack.separatrix](#48-separatrix--lcfstracing)
   - [gspack.safety](#49-safety--安全因子q)
   - [gspack.diagnostics](#410-diagnostics--诊断工具)
   - [gspack.machine](#411-machine--线圈与机器定义)
   - [gspack.control](#412-control--等离子体控制)
   - [gspack.io](#413-io--hdf5-存档)
   - [gspack.geqdsk](#414-geqdsk--g-eqdsk-读写)
   - [gspack.optimize](#415-optimize--形状优化与uq)
5. [示例详解](#5-示例详解)
6. [数据集生成](#6-数据集生成)
7. [常见问题](#7-常见问题)

---

## 1. 物理模型

### 1.1 Grad-Shafranov 方程

Grad-Shafranov 方程描述了轴对称托卡马克等离子体平衡：

$$\Delta^*\psi = -\mu_0 R J_\phi(R,Z)$$

其中：
- $\psi(R,Z)$ — **极向磁通函数**，满足 $B_p = \frac{1}{R}\nabla\psi \times \hat{\phi}$
- $J_\phi(R,Z)$ — **环向电流密度**
- 等离子体压力 $p(\psi)$ 和极向电流函数 $f(\psi) = R B_\phi$ 只是 $\psi$ 的函数

等价形式：

$$\Delta^*\psi = -\mu_0 R^2 p'(\psi) - f f'(\psi)$$

其中 $p' = dp/d\psi$，$ff' = f \cdot df/d\psi$。

**边界条件**：
- **自由边界**：矩形计算域边界上，$\psi$ 由线圈电流的 Green 函数贡献 + 等离子体的
  von Hagenow 边界积分确定
- **固定边界**：在预设的 D 形 LCFS 轮廓上，$\psi = \psi_{\text{bndry}}$（常数，通量面条件）

### 1.2 电流剖面参数化（Jeon 2015）

电流剖面采用 Jeon et al. (Nuclear Fusion, 2015) 提出的双参数形式：

$$J_\phi = L \left[\beta_0 \frac{R}{R_0} + (1-\beta_0)\frac{R_0}{R}\right]
\left(1 - \hat{\psi}^{\alpha_m}\right)^{\alpha_n}$$

其中：
- $\hat{\psi} = (\psi - \psi_{\text{axis}})/(\psi_{\text{bndry}} - \psi_{\text{axis}})$
  是归一化极向磁通函数
- $\alpha_m, \alpha_n$ — **电流剖面指数**，控制电流分布形状
- $\beta_0$ — **电流剖面参数**，控制 Bootstrap 电流比例
- $L$ — **归一化系数**

两个线性参数 $L$ 和 $\beta_0$ 通过两个全局约束确定：

**约束 1：总等离子体电流**
$$I_p = \int J_\phi \, dR \, dZ$$

**约束 2：极向比压**（`ConstrainBetapIp` 时）
$$\beta_p = \frac{2\mu_0 \langle p \rangle}{\langle B_p^2 \rangle}$$

或 **轴心压力**（`ConstrainPaxisIp` 时）
$$p_0 = p(\psi_{\text{axis}})$$

`ConstrainBetapIp` 直接给出通量面平均极向比压约束下的电流分布，适用于实验
数据驱动的计算。`ConstrainPaxisIp` 直接控制轴心压力，更适合理论设计。

### 1.3 Green 函数法

单位环向电流丝在 $(R', Z')$ 处产生的极向磁通为：

$$G(R,Z; R',Z') = \frac{\mu_0}{2\pi} \sqrt{\frac{R R'}{k}}
\left[(2-k^2)K(k) - 2E(k)\right]$$

其中：
$$k^2 = \frac{4RR'}{(R+R')^2 + (Z-Z')^2}$$

$K(k)$ 和 $E(k)$ 分别是第一类和第二类完全椭圆积分。

任意电流分布 $J_\phi(R',Z')$ 产生的极向磁通：

$$\psi(R,Z) = \iint G(R,Z; R',Z') \cdot J_\phi(R',Z') \, dR' \, dZ'$$

这是**自由空间解**——自动满足真空区 Laplace 方程，且 $\psi \to 0$ 在无穷远处。

### 1.4 von Hagenow 自由边界方法

自由边界求解器使用 **von Hagenow 方法**，是一个两步迭代：

1. **固定边界求解**：在矩形域上解 $\Delta^*\psi = -\mu_0 R J_\phi$，$\psi|_{\partial\Omega}=0$
2. **边界积分**：用 Green 第二恒等式，通过边界法向导数计算真实边界 $\psi$ 值：
   $$\psi(R_{\text{bndry}}) = \oint_{\partial\Omega} \left[G\frac{\partial\psi}{\partial n} - \psi\frac{\partial G}{\partial n}\right] \, dl$$
3. **重新求解**：以新边界值再解 GS 方程

### 1.5 自洽固定边界方法

固定边界求解器使用 **自洽的 Green 函数边界条件**：

1. 从当前 $\psi$ 计算 $J_\phi$（利用预设的电流剖面约束 $I_p, \beta_p$）
2. 在 D 形轮廓上计算 Green 体积分：
   $$\psi_{\text{green}}(R_{\text{LCFS}}, Z_{\text{LCFS}}) = \iint G \cdot J_\phi \, dR' \, dZ'$$
3. D 形轮廓上 $\psi$ 取平均值作为 LCFS 通量面值：
   $$\psi_{\text{bndry}} = \langle \psi_{\text{green}} \rangle_{\text{LCFS}}$$
4. 以 $\psi = \psi_{\text{bndry}}$ 为 Dirichlet BC，在 D 形内部求解 GS 方程
5. 迭代至收敛后，Green 体积分在 D 形上的结果 $\approx \psi_{\text{bndry}}$ — 自洽

该方法的关键优势是收敛后可以直接用 Green 体积分（`psi_on_grid()`）扩展到任意
外部网格计算真空区 $\psi$ 和 $B_p$，结果天然自洽。

---

## 2. 安装与依赖

### 核心依赖

```
numpy
scipy
matplotlib
```

### 可选依赖

| 包 | 用途 | 安装 |
|---|---|---|
| `cupy-cuda12x` | GPU 加速 | `pip install cupy-cuda12x` |
| `h5py` | 数据存档 | `pip install h5py` |
| `pyamg` | 代数多重网格求解器（大网格） | `pip install pyamg` |
| `tqdm` | 数据集生成进度条 | `pip install tqdm` |

### 安装

```bash
# 克隆仓库后无需安装，直接在项目根目录运行
cd gspack2_TRAE
python examples/01_freeboundary.py
```

GPU 模式自动切换（若 CuPy 可用）：

```python
import gspack.backend as bk
bk.set_backend('gpu')   # 或 'auto' 自动检测
```

---

## 3. 快速开始

### 3.1 自由边界求解

```python
import numpy as np
from gspack.machine import TestTokamak
from gspack.equilibrium import Equilibrium
from gspack.profiles import ConstrainPaxisIp
from gspack import picard

# 1. 创建机器（含线圈）
tok = TestTokamak()

# 2. 创建平衡对象（65×65 网格）
eq = Equilibrium(tok, nx=65, ny=65)

# 3. 创建电流剖面（固定 Ip = 200 kA, p_axis = 10 kPa）
pro = ConstrainPaxisIp(p_axis=1e4, Ip=2e5, fvac=1.0, Raxis=1.0)

# 4. Picard 迭代求解
picard.solve(eq, pro, maxits=50, rtol=1e-3)

# 5. 诊断
print(f"Ip      = {eq.plasmaCurrent():.2e} A")
print(f"βp      = {eq.poloidalBeta():.4f}")
print(f"ψ_axis  = {eq.psi_axis:.6f} Wb/rad")
print(f"q(0.95) = {eq.q(np.array([0.95]))[0]:.3f}")

# 6. 磁场
Br = eq.Br(1.0, 0.0)
Bz = eq.Bz(1.0, 0.0)
Btor = eq.Btor(1.0, 0.0)
```

### 3.2 固定边界求解

```python
from gspack.equilibrium import FixedBoundaryEquilibrium
from gspack.profiles import ConstrainBetapIp
from gspack import picard

# 1. 预设 D 形 LCFS（R₀=1.0 m, a=0.5 m, κ=1.6, δ=0.33）
eq_fb = FixedBoundaryEquilibrium(
    R0=1.0, a=0.5, kappa=1.6, delta=0.33,
    nx=65, ny=65)

# 2. 电流剖面（固定 Ip = 200 kA, βp = 0.8）
pro_fb = ConstrainBetapIp(
    betap=0.8, Ip=2e5, fvac=1.0,
    alpha_m=1.0, alpha_n=2.0, Raxis=1.0)

# 3. 求解（constrain=None：不使用线圈）
picard.solve(eq_fb, pro_fb, constrain=None, rtol=1e-3)

# 4. 结果
print(f"Ip    = {eq_fb.plasmaCurrent():.2e} A")
print(f"βp    = {eq_fb.poloidalBeta():.4f}")
print(f"ψ_a   = {eq_fb.psi_axis:.4f} Wb/rad")
print(f"ψ_b   = {eq_fb.psi_bndry:.4f} Wb/rad")
```

### 3.3 外部区域求解（Green 体积分）

固定边界求解收敛后，使用 Green 体积分在任意大网格上计算真空区 $\psi$：

```python
import numpy as np
from gspack.backend import set_backend

set_backend('cpu')

# 假设 eq_fb 已收敛
R_v = np.linspace(0.1, 2.5, 97)
Z_v = np.linspace(-1.2, 1.2, 97)
R_v2d, Z_v2d = np.meshgrid(R_v, Z_v, indexing='ij')

psi_v = eq_fb.psi_on_grid(R_v2d, Z_v2d)

# 计算真空区极向场
from scipy.interpolate import RectBivariateSpline
f_psi = RectBivariateSpline(R_v, Z_v, psi_v)
Br_v = -f_psi(R_v, Z_v, dy=1, grid=True) / R_v2d
Bz_v =  f_psi(R_v, Z_v, dx=1, grid=True) / R_v2d
```

`psi_on_grid()` 返回的是**自由空间 Green 体积分**，在 D 形外部自动满足
Laplace 方程，且边界处自洽（$\psi|_{\text{LCFS}} \approx \psi_{\text{bndry}}$）。

---

## 4. API 参考

### 4.1 `gspack.backend` — CPU/GPU 后端

透明地在 NumPy 和 CuPy 之间切换。

```python
import gspack.backend as bk

bk.set_backend('cpu')     # 强制 CPU（默认）
bk.set_backend('gpu')     # 强制 GPU（需 CuPy）
bk.set_backend('auto')    # 自动检测，优先 GPU
bk.get_backend()          # → 'cpu' 或 'gpu'
```

**关键函数**：

| 函数 | 说明 |
|------|------|
| `get_xp()` | 返回当前活跃的数组模块（`np` 或 `cp`） |
| `to_numpy(arr)` | 将任意数组转为 NumPy ndarray |
| `to_backend(arr)` | 将数组移到当前设备 |
| `asarray(x)` | 将标量/数组转为当前设备 ndarray |
| `ellipk_compat(k2)` | $K(k^2)$ — 兼容 GPU |
| `ellipe_compat(k2)` | $E(k^2)$ — 兼容 GPU |

**常量**：
- `MU0 = 4π × 10⁻⁷` — 真空磁导率

**设计规则**：
- 不要模块级 `from .backend import xp`（`set_backend` 后不更新）
- 始终在函数内调用 `get_xp()`

### 4.2 `gspack.greens` — Green 函数与 GS 算子

**Green 函数**：

```python
from gspack.greens import greens, greens_Br, greens_Bz, MU0

# G(Rc,Zc; R,Z) — 单位电流丝在 (Rc,Zc) 处产生的 ψ
g = greens(Rc, Zc, R, Z)

# 磁场分量（中心差分）
Br = greens_Br(Rc, Zc, R, Z)
Bz = greens_Bz(Rc, Zc, R, Z)
```

**GS 稀疏矩阵**：

```python
from gspack.greens import gs_sparse, gs_sparse_2nd, gs_sparse_4th

# 二阶中心差分 A_2nd (CSR)
A2 = gs_sparse_2nd(Rmin, Rmax, Zmin, Zmax, nx, ny)

# 四阶中心差分 A_4th (LIL)
A4 = gs_sparse_4th(Rmin, Rmax, Zmin, Zmax, nx, ny)
```

**求解器工厂**：

```python
from gspack.greens import make_solver, make_masked_solver

# 标准矩形域求解器
solver = make_solver(Rmin, Rmax, Zmin, Zmax, nx, ny,
                     order=2, method='auto')
# 返回 callable solve(rhs_2d) → psi_2d

# 掩码求解器（任意内部/边界）
solver_mask = make_masked_solver(
    Rmin, Rmax, Zmin, Zmax, nx, ny,
    interior_mask,  # (nx, ny) bool: True=GS模板, False=Dirichlet BC
    order=2, method='lu')
```

### 4.3 `gspack.equilibrium` — 平衡类

#### `Equilibrium` — 自由边界平衡

```python
from gspack.equilibrium import Equilibrium

eq = Equilibrium(tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                  nx=65, ny=65, order=2, method='auto', check_limited=False)
```

**属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `R, Z` | (nx, ny) ndarray | 计算网格 |
| `dR, dZ` | float | 网格间距 |
| `psi_axis` | float | 轴磁通 |
| `psi_bndry` | float | 边界磁通 |
| `plasma_psi` | (nx, ny) ndarray | 等离子体 $\psi$（不含线圈） |
| `_Jtor` | (nx, ny) ndarray | 环向电流密度 |
| `_opoints` | list | O 点列表 |
| `_xpoints` | list | X 点列表 |

**方法**：

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `psi()` | (nx, ny) ndarray | 总 $\psi$（含线圈） |
| `psiN()` | (nx, ny) ndarray | 归一化 $\hat{\psi}$ |
| `psiRZ(R, Z)` | float | 任意点 $\psi$ 插值 |
| `Br(R, Z)` | ndarray | 极向场 $B_R$ |
| `Bz(R, Z)` | ndarray | 极向场 $B_Z$ |
| `Bpol(R, Z)` | ndarray | $|B_p|$ |
| `Btor(R, Z)` | ndarray | $B_\phi$ |
| `solve(profiles, psi=None)` | None | Picard 迭代步 |
| `plasmaCurrent()` | float | $I_p$ |
| `poloidalBeta()` | float | $\beta_p$ |
| `toroidalBeta()` | float | $\beta_t$ |
| `betaN()` | float | 归一化 $\beta$ |
| `internalInductance()` | float | $l_i$ |
| `q(psiN_arr)` | ndarray | 安全因子 |
| `magneticAxis()` | (R, Z, ψ) | 磁轴 |
| `separatrix(npoints)` | (N, 2) ndarray | LCFS 追踪 |
| `geometricAxis()` | (R, Z) | 几何轴 |
| `shafranovShift()` | (dR, dZ) | Shafranov 偏移 |
| `elongation()` | float | 拉长比 $\kappa$ |
| `triangularity()` | float | 三角形变 $\delta$ |
| `aspectRatio()` | float | 纵横比 $R/a$ |
| `plasmaVolume()` | float | 等离子体体积 |

#### `FixedBoundaryEquilibrium` — 固定边界平衡

```python
from gspack.equilibrium import FixedBoundaryEquilibrium

eq = FixedBoundaryEquilibrium(
    R0=1.0, a=0.5, kappa=1.6, delta=0.33,
    Rmin=0.2, Rmax=1.8, Zmin=-0.8, Zmax=0.8,
    nx=65, ny=65, order=2, method='auto')
```

**额外属性**：

| 属性 | 说明 |
|------|------|
| `R0, a, kappa, delta` | D 形 LCFS 参数 |
| `R_lcfs, Z_lcfs` | (360,) LCFS 轮廓坐标 |
| `plasma_mask` | (nx, ny) bool — D 形内部掩码 |

**额外方法**：

```python
# 在任意网格上计算 ψ（Green 体积分，用于外部真空区）
psi_ext = eq.psi_on_grid(R_obs, Z_obs)
```

### 4.4 `gspack.boundary` — 边界条件

**自由边界**：

```python
from gspack.boundary import free_boundary_hagenow

psi = free_boundary_hagenow(R, Z, Jtor, solver)
# von Hagenow 方法：先固定边界解 → 法向导数 → Green 边界积分 → 重新解
```

**固定边界**：

```python
from gspack.boundary import fixed_boundary_solve

psi = fixed_boundary_solve(R, Z, Jtor, solver)
# 第一步同 von Hagenow，但不重新迭代线圈
```

**Green 体积分**：

```python
from gspack.boundary import greens_volume_psi

psi_obs = greens_volume_psi(R_obs, Z_obs, R_src, Z_src, Jtor_src, dR, dZ)
# 自由空间 ψ = ∫∫ G·Jφ dS
```

**D 形 LCFS 工具**：

```python
from gspack.boundary import dshape_lcfs, mask_inside_lcfs, initial_psi_lcfs

# 生成 D 形轮廓
R_lcfs, Z_lcfs = dshape_lcfs(R0=1.0, a=0.5, kappa=1.6, delta=0.33, ntheta=360)

# 生成 D 形内部掩码
mask = mask_inside_lcfs(R_grid, Z_grid, R_lcfs, Z_lcfs)

# 生成抛物型初猜
psi0 = initial_psi_lcfs(R_grid, Z_grid, R_lcfs, Z_lcfs,
                         psi_axis=1.0, psi_bndry=0.0)
```

### 4.5 `gspack.profiles` — 等离子体电流剖面

所有剖面类共享的电流形式：

$$J_\phi = L \left[\beta_0 \frac{R}{R_0} + (1-\beta_0)\frac{R_0}{R}\right]
(1 - \hat{\psi}^{\alpha_m})^{\alpha_n}$$

#### `_ProfileBase` — 基类

```python
class _ProfileBase:
    alpha_m: float     # 电流剖面指数 1 (default 1.0)
    alpha_n: float     # 电流剖面指数 2 (default 2.0)
    Raxis: float       # 参考大半径 (default 1.0)

    def _shape(self, psiN)         # → (1 - ψN^αm)^αn
    def pprime(self, psiN)         # p'(ψN)
    def ffprime(self, psiN)        # ff'(ψN)
    def fvac(self)                 # R_0 · B_φ0 (vacuum field)
    def pressure(self, psiN)       # p(ψN)  by integration
    def fpol(self, psiN)           # f(ψN)  by integration
```

#### `ConstrainPaxisIp` — 固定轴心压力 + Ip

```python
pro = ConstrainPaxisIp(p_axis=1e4,     # 轴心压力 [Pa]
                        Ip=2e5,         # 总电流 [A]
                        fvac=1.0,       # 真空 R·Bφ [T·m]
                        alpha_m=1.0,    # 电流剖面指数
                        alpha_n=2.0,
                        Raxis=1.0)
```

#### `ConstrainBetapIp` — 固定极向比压 + Ip

```python
pro = ConstrainBetapIp(betap=0.8,       # 极向比压
                        Ip=2e5,          # 总电流 [A]
                        fvac=1.0,
                        alpha_m=1.0,
                        alpha_n=2.0,
                        Raxis=1.0)
```

#### `ConstrainRotation` — 固定 p₀ + Ip + 环向旋转

```python
pro = ConstrainRotation(p_axis=1e4,
                         Ip=2e5,
                         fvac=1.0,
                         omega_profile=lambda psiN: ...,
                         rho_profile=lambda psiN: ...,
                         alpha_m=1.0, alpha_n=2.0, Raxis=1.0)
```

添加离心力项 $J_{\text{rot}} = \rho\Omega^2 R$。

#### `ProfilesPprimeFfprime` — 任意指定 p' 和 ff'

```python
pro = ProfilesPprimeFfprime(
    pprime_fn=lambda psiN: ...,      # p'(ψN)
    ffprime_fn=lambda psiN: ...,     # ff'(ψN)
    fvac=1.0)
```

### 4.6 `gspack.picard` — Picard 迭代

```python
from gspack import picard

errs = picard.solve(eq, profiles,
                     constrain=constrain_obj,  # 或 None（固定边界）
                     maxits=50,
                     rtol=1e-3,
                     atol=1e-10,
                     anderson_m=5,     # 0 = 无加速
                     verbose=True)
```

**Anderson 加速**：

`AndersonMixer(m=5)` 使用历史窗口 $m$ 的 Anderson 混合：

$$\psi^{(k+1)} = \sum_{i=0}^{m-1} c_i (\psi^{(k-i)} + r^{(k-i)})$$

其中 $r = \psi_{\text{new}} - \psi_{\text{old}}$ 是残差，系数 $c_i$ 通过最小
二乘确定。相比纯 Picard 迭代提供约 3 倍收敛加速。

**返回值**：当 `convergenceInfo=True` 时返回 `(max_change_array, rel_change_array)`。

### 4.7 `gspack.critical` — O 点和 X 点检测

```python
from gspack import critical

opoints, xpoints = critical.find_critical(R, Z, psi)
# opoints: list of (R, Z, psi) — O 点
# xpoints: list of (R, Z, psi) — X 点

mask = critical.core_mask(R, Z, psi, opoints, xpoints, psi_bndry)
# core_mask: 等离子体核心掩码（flood-fill 从 O 点开始）

psi_axis, psi_bndry, opoints, xpoints = \
    critical.update_psi_boundary(R, Z, psi)
```

**算法**：
1. 计算 $|\nabla\psi|^2$，找局部极小值作为候选零场点
2. Newton-Raphson 迭代精化到 $B_R = B_Z = 0$
3. 用 Hessian 判别式 $S = \psi_{RR}\psi_{ZZ} - \psi_{RZ}^2$ 区分 O 点 ($S>0$) 和 X 点 ($S<0$)
4. 对 X 点做单调性滤波（从 O 点沿径向到 X 点的 $\psi$ 必须单调）

### 4.8 `gspack.separatrix` — LCFS 追踪

```python
from gspack.separatrix import find_separatrix

sep = find_separatrix(eq, ntheta=360)
# sep: (ntheta, 2) ndarray — (R, Z) of LCFS
```

**算法**：对每个极向角 $\theta$，从 O 点沿射线二分查找 $\psi_N = 1$ 的位置。
最大步长限制在 O 点到主 X 点距离的 1.05 倍，避免穿过 X 点进入 SOL 区。

### 4.9 `gspack.safety` — 安全因子 $q$

```python
from gspack.safety import find_safety

q_arr = find_safety(eq, psiN_arr, ntheta=128)
# q_arr: 与 psiN_arr 同长

# 或直接通过 Equilibrium 调用
q = eq.q(np.array([0.01, 0.25, 0.50, 0.75, 0.95]))
```

**算法**：对每个通量面追踪其 $(R(\theta), Z(\theta))$ 轮廓，然后计算：

$$q = \frac{f}{2\pi} \oint \frac{dl}{R B_p}$$

### 4.10 `gspack.diagnostics` — 诊断工具

```python
from gspack import diagnostics

# Poincaré 截面
traces = diagnostics.poincare_section(
    eq, R0_list=[1.1, 1.2], Z0_list=[0.0, 0.0],
    n_turns=200, n_steps_per_turn=200)

# 通量面坐标
fs_coords = diagnostics.flux_surface_coordinates(
    eq, n_psi=50, n_theta=128)
# 返回: psiN, theta, R, Z, B, Bpol, q, dV_dpsiN

# MHD 稳定性指标
mhd = diagnostics.mhd_stability_indicators(eq)
# 返回: q_axis, q_95, 有理面位置, betaN, Troyon_margin, li, ...
```

### 4.11 `gspack.machine` — 线圈与机器定义

```python
from gspack.machine import Coil, ShapedCoil, Machine, TestTokamak, Wall

# 圆环线圈
c = Coil(R=1.0, Z=0.5, current=1e6, turns=10, control=True)
psi = c.psi(R_grid, Z_grid)
Br  = c.Br(R_grid, Z_grid)

# 矩形截面线圈
sc = ShapedCoil(corners=[(1.0, 0.4), (1.0, 0.6), (1.2, 0.6), (1.2, 0.4)],
                 current=1e6, turns=10)

# 预置测试托卡马克（TestTokamak）
from gspack.machine import TestTokamak
tok = TestTokamak()
# 包含 PF1-PF6 六组极向场线圈
```

### 4.12 `gspack.control` — 等离子体控制

```python
from gspack.control import constrain, constrain_snowflake

# X 点 + 等磁通约束
ctrl = constrain(xpoints=[(1.2, -0.4), (1.2, 0.4)],
                 isoflux=[(1.0, 0.5, 1.0, -0.5)],
                 psivals=[(1.0, 0.0, 0.0)],
                 gamma=1e-12)

# 在 Picard 迭代中使用
picard.solve(eq, pro, constrain=ctrl)
```

使用 Tikhonov 正则化最小二乘：$\Delta I = (A^T A + \gamma^2 I)^{-1} A^T b$

### 4.13 `gspack.io` — HDF5 存档

```python
from gspack import io

# 保存
io.save(eq, "equilibrium.h5")

# 读取（手动处理 h5py）
import h5py
with h5py.File("equilibrium.h5", 'r') as f:
    psi = f['fields/psi'][:]
    R = f['grid/R'][:]
    Z = f['grid/Z'][:]
```

HDF5 文件结构：

```
/grid/R, /grid/Z                  — 网格
/fields/psi, /fields/plasma_psi   — ψ 场
/fields/Jtor                      — 电流密度
/fields/Bpol                      — 极向场
/scalars/psi_axis, /scalars/psi_bndry  — 标量
/scalars/Ip, betap, kappa, delta  — 全局量
/profiles/psiN_1d, q, pressure, fpol — 剖面
/separatrix/R, /separatrix/Z      — LCFS
/coils/                           — 线圈电流
/opoints                          — O 点
```

### 4.14 `gspack.geqdsk` — G-EQDSK 读写

```python
from gspack import geqdsk

# 写入（COCOS 1 约定）
geqdsk.write_geqdsk(eq, "output.geqdsk", shot=12345, time=1.0)

# 读取
data = geqdsk.read_geqdsk("output.geqdsk")
# data.psi, data.R, data.Z, data.fpol, data.pres, data.qpsi, ...
```

兼容 TRANSP、VMEC、GENE、BOUT++、OMFIT 等常用代码。

### 4.15 `gspack.optimize` — 形状优化与 UQ

```python
from gspack import optimize

# 线圈电流优化（目标 κ=1.6, δ=0.33, R_mag=1.25）
result = optimize.optimize_shape(
    eq, profiles, constrain_obj=ctrl,
    target_kappa=1.6, target_delta=0.33, target_R_mag=1.25,
    weights=(1.0, 1.0, 1.0), method='Nelder-Mead',
    maxits_picard=10, maxits_opt=200)
```

---

## 5. 示例详解

### `examples/01_freeboundary.py`

自由边界 GS 平衡求解，使用 TestTokamak 的 6 组 PF 线圈。

关键步骤：
1. 创建机器对象 `TestTokamak()`
2. 创建 `Equilibrium` 和 `ConstrainPaxisIp`
3. `picard.solve()` 迭代至收敛
4. 绘制通量面、$q$ 剖面和收敛曲线

### `examples/02_fixedboundary.py`

固定边界 GS 平衡求解，预设 D 形 LCFS。

关键特性：
- 使用 `FixedBoundaryEquilibrium`，$R_0=1.0, a=0.5, \kappa=1.6, \delta=0.33$
- 使用 `ConstrainBetapIp`，$I_p=200 \text{ kA}, \beta_p=0.8$
- 自洽边界条件：$\psi_{\text{bndry}}$ 从 Green 体积分自动确定
- 收敛后打印自洽性检验

### `examples/03_fixedboundary_largegrid.py`

固定边界求解 + 外部真空区 Green 体积分。

关键特性：
- 求解 D 形内部 FDM 解（用于等离子体诊断）
- 在大矩形网格上用 `psi_on_grid()` 计算外部 $\psi$
- 计算容器级极向场 $B_p$

### `examples/04_generate_dataset.py`

数据集生成脚本，使用 Latin Hypercube Sampling 对 8 个参数采样。

参数范围：

| 参数 | 范围 | 说明 |
|------|------|------|
| $R_0$ | [0.8, 1.5] m | 大半径 |
| $a$ | [0.3, 0.7] m | 小半径 |
| $\kappa$ | [1.0, 2.0] | 拉长比 |
| $\delta$ | [0.0, 0.5] | 三角形变 |
| $I_p$ | [1e5, 5e5] A | 等离子体电流 |
| $\beta_p$ | [0.3, 1.5] | 极向比压 |
| $\alpha_m$ | [0.5, 3.0] | 电流剖面指数 |
| $\alpha_n$ | [0.5, 3.0] | 电流剖面指数 |

输出为 HDF5 格式，每个样本存储网格、$\psi$ 分布和收敛信息。

### `examples/05_visualize_dataset.py`

数据集可视化脚本，从 HDF5 中随机抽样并绘制 $\psi$ 等值线图。

```bash
python examples/05_visualize_dataset.py
python examples/05_visualize_dataset.py --nsamples 12 --nrows 3 --ncols 4
```

---

## 6. 数据集生成

### 批量生成

```bash
# 生成 100 个样本
python examples/04_generate_dataset.py --nsamples 100

# 生成 500 个样本（使用 tqdm 进度条）
python examples/04_generate_dataset.py --nsamples 500

# 从已有数据集继续
python examples/04_generate_dataset.py --resume datasets/gs_dataset_xxx.h5 --nsamples 200

# 快速查看数据集信息
python examples/04_generate_dataset.py --quick
```

### 性能参考

| 网格 | 每样本时间 | 1000 样本 |
|------|-----------|-----------|
| 33×33 | ~0.5 s | ~8 min |
| 65×65 | ~1.5 s | ~25 min |
| 129×129 | ~6 s | ~100 min |

### HDF5 数据结构

```
/parameters        (N, 8)  float32 — 参数表
/parameter_names   (8,)    S       — 参数名
/converged         (N,)    bool    — 收敛标志
/niter             (N,)    int32   — 迭代次数
/psi_axis          (N,)    float32
/psi_bndry         (N,)    float32
/nx, /ny           (N,)    int32   — 各样本网格尺寸
/samples/000000/
    R_grid         (nx,)   float32 — R 坐标
    Z_grid         (ny,)   float32 — Z 坐标
    psi            (nx,ny) float32 — ψ 分布
/samples/000001/
    ...
```

---

## 7. 常见问题

### Q: 网格分辨率如何选择？

- **65×65**（默认）— 快速开发，~1.5 s/次固定边界求解
- **129×129** — 平衡速度和精度，~6 s/次
- **257×257** — 精细计算（建议搭配 pyamg 加速）

网格尺寸建议使用 $2^n + 1$ 以确保 Romberg 积分兼容。

### Q: FDM 解和 Green 体积分有什么区别？

- `eq.psi()` — FDM 解，在 D 形上 $\psi = \psi_{\text{bndry}}$（Dirichlet BC）D 形内部精确
  满足 GS 方程，D 形外部无物理意义（被矩形 BC 约束）
- `eq.psi_on_grid()` — Green 体积分，自由空间解，在 D 形上 $\psi \approx \psi_{\text{bndry}}$
  （自洽收敛），在真空区自动满足 Laplace 方程

两者在 D 形内部有量级约 $O(\Delta R^2 + \Delta Z^2)$ 的离散误差，网格加密可减小。

### Q: 求解不收敛怎么办？

1. 增加 `maxits`（默认 50）
2. 调低 `rtol`（默认 1e-3，可试 5e-3）
3. 增加 Anderson 窗口 `anderson_m`（默认 5，可试 8）
4. 检查参数是否在合理范围内（特别是 $\alpha_m, \alpha_n$ 过大时剖面趋近阶跃函数）
5. 尝试更粗的网格（33×33）先验证参数组合

### Q: 怎么添加新线圈？

```python
from gspack.machine import Coil, Machine

m = Machine()
m.addCoil("PF1", Coil(R=0.5, Z=0.8, current=0.0, turns=100))
m.addCoil("PF2", Coil(R=0.5, Z=-0.8, current=0.0, turns=100))
```

### Q: GPU 加速有效吗？

对于 129×129 以上网格，CuPy 加速 Green 函数计算效果显著（2-5 倍）。但稀疏
矩阵求解始终在 CPU（SciPy 要求），所以整体加速比受限于矩阵求解部分。

---

## 参考文献

1. **Jeon, Y.M. (2015)**. "Development of a free boundary Tokamak
   Equilibrium Solver (TES) for Advanced Study of Tokamak Equilibria."
   *Journal of the Korean Physical Society*, 67(5), 843-853.

2. **Cerfon, A.J. & Solovev, I.V. (2012)**. "One-size-fits-all analytic
   solutions to the Grad–Shafranov equation."
   *Physics of Plasmas*, 19, 032506.

3. **von Hagenow, K. (1969)**. "A method for the solution of the
   free-boundary magnetohydrodynamic equilibrium equation."
   *IPP 6/106*, Max-Planck-Institut für Plasmaphysik.

4. **Dudson, B. (2016)**. FreeGS: Free boundary Grad-Shafranov solver.
   GitHub: https://github.com/freegs/freegs
