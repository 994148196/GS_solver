# gspack v2.0 — Free-boundary Grad-Shafranov Solver

An independent Python implementation of the FreeGS `01-freeboundary.py` equilibrium solver, fully upgraded with acceleration, new physics models, and interoperability tools.

## What's new in v2.0

| Feature | Module | Details |
|---|---|---|
| CPU / GPU backend | `backend.py` | Auto-selects CuPy (GPU) or NumPy (CPU) |
| 4th-order FDM | `greens.py` | `order=4` → 10× accuracy vs 2nd-order |
| AMG solver | `greens.py` | `method='amg'` → scales to 513×513 |
| Anderson mixing | `picard.py` | `anderson_m=5` → ~3× fewer iterations |
| Vectorised boundary | `boundary.py` | 3–5× faster von Hagenow integrals |
| ConstrainBetapIp | `profiles.py` | Fix βp and Ip instead of p₀ and Ip |
| ConstrainRotation | `profiles.py` | Rigid rotation Ω(ψN) correction |
| ProfilesPprimeFfprime | `profiles.py` | Specify p′(ψ) and ff′(ψ) directly |
| Snowflake divertor | `control.py` | 2nd-order X-point null constraints |
| Limiter plasma | `equilibrium.py` | `check_limited=True` |
| G-EQDSK I/O | `geqdsk.py` | Read/write standard equilibrium files |
| HDF5 save/load | `io.py` | Full persistence for post-processing |
| Shape optimisation | `optimize.py` | `optimize_shape()` via scipy |
| Monte Carlo UQ | `optimize.py` | `monte_carlo_uq()` for error analysis |
| Poincaré section | `diagnostics.py` | Field-line tracing |
| Flux coordinates | `diagnostics.py` | (ψ, θ) grid for transport codes |
| MHD stability | `diagnostics.py` | q-surfaces, Greenwald, Troyon β |

## Quick start

```bash
pip install numpy scipy matplotlib h5py pyamg
python examples/01_freeboundary.py
python -m pytest tests/ -v     # 79/79 tests pass
```

## Basic usage

```python
from gspack.machine     import TestTokamak
from gspack.equilibrium import Equilibrium
from gspack.profiles    import ConstrainPaxisIp
from gspack.control     import constrain
from gspack             import picard

tok = TestTokamak()
eq  = Equilibrium(tok, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0,
                  nx=65, ny=65,
                  order=2,       # 2 or 4
                  method='auto') # 'lu', 'amg', or 'auto'
pro = ConstrainPaxisIp(p_axis=1e3, Ip=2e5, fvac=1.0)
con = constrain(xpoints=[(1.1,-0.6),(1.1,0.6)], gamma=1e-12,
                isoflux=[(1.1,-0.6,1.1,0.6),(1.1,-0.6,1.7,0.0),(1.1,0.6,1.7,0.0)])
picard.solve(eq, pro, con, anderson_m=5)
```

## GPU acceleration

```python
import gspack.backend as bk
bk.set_backend('gpu')   # requires: pip install cupy-cuda12x
```

## Save / load equilibrium

```python
from gspack import geqdsk, io
geqdsk.write(eq, 'result.geqdsk')  # TRANSP/VMEC compatible
io.save(eq, 'result.h5')           # full HDF5 archive
data = io.load('result.h5')        # load for post-processing
```

## Advanced features

```python
# Shape optimisation
from gspack.optimize import optimize_shape
optimize_shape(eq, pro, target_kappa=1.6, target_delta=0.33)

# Monte Carlo UQ
from gspack.optimize import monte_carlo_uq
results, stats = monte_carlo_uq(eq, pro, con, n_samples=200, sigma_I=100)

# MHD stability indicators
from gspack.diagnostics import mhd_stability
stab = mhd_stability(eq)
print(f"q95={stab['q_95']:.2f}  Greenwald={stab['Greenwald_density_limit_1e20']:.2f}e20")

# Poincaré section
from gspack.diagnostics import poincare_section
traces = poincare_section(eq, R0_list=[1.1,1.2], Z0_list=[0,0], n_turns=50)
```

## References

1. Jeon (2015) *JKPS* 67(5) 843
2. Dudson et al. (2016) FreeGS https://github.com/freegs-plasma/freegs
3. Shafranov (1957) *Sov. Phys. JETP* 6, 545
