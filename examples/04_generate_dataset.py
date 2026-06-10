"""
gspack v2.0 — 04-generate-dataset
==================================
Sample 8 parameters → ψ(R,Z) mapping dataset via Latin Hypercube Sampling.

8 parameters:
  R₀, a, κ, δ    — D-shaped LCFS (Cerfon–Solovev)
  I_p, β_p       — plasma current & poloidal beta constraints
  α_m, α_n       — current profile exponents (Jeon 2015 Eq.5)

Output: HDF5 file with:
  /parameters   — (N, 8) parameter table  [R0, a, kappa, delta, Ip, betap, alpha_m, alpha_n]
  /grid_R       — (N, nx,)  R grid per sample  (or /grid_R0 shared if domain is fixed)
  /grid_Z       — (N, ny,)  Z grid per sample
  /psi          — (N, nx, ny)  ψ fields
  /psi_axis     — (N,)  axis values
  /psi_bndry    — (N,)  boundary values
  /converged    — (N,)  bool convergence flag
  /niter        — (N,)  iteration count

Usage:
  python examples/04_generate_dataset.py                   # default: 100 samples
  python examples/04_generate_dataset.py --nsamples 500    # 500 samples
  python examples/04_generate_dataset.py --resume           # resume from checkpoint
"""

import sys, os, warnings, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from gspack.equilibrium import FixedBoundaryEquilibrium
from gspack.profiles    import ConstrainBetapIp
from gspack             import picard
import gspack.backend as bk

# ─────────────────────────────────────────────────────────────────────────────
#  Parameter bounds
# ─────────────────────────────────────────────────────────────────────────────
PARAM_BOUNDS = {
    "R0":     (0.8,  1.5),    # major radius [m]
    "a":      (0.3,  0.7),    # minor radius [m]
    "kappa":  (1.0,  2.0),    # elongation
    "delta":  (0.0,  0.5),    # triangularity
    "Ip":     (1e5,  5e5),    # plasma current [A]
    "betap":  (0.3,  1.5),    # poloidal beta
    "alpha_m":(0.5,  3.0),    # current profile exponent 1
    "alpha_n":(0.5,  3.0),    # current profile exponent 2
}
PARAM_NAMES = ["R0", "a", "kappa", "delta", "Ip", "betap", "alpha_m", "alpha_n"]
N_PARAMS = len(PARAM_NAMES)

# Grid settings
NX, NY    = 65, 65
ORDER     = 2
METHOD    = "lu"
ANDERSON_M = 5
MAXITS    = 50
RTOL      = 5e-3


# ─────────────────────────────────────────────────────────────────────────────
#  Sampling
# ─────────────────────────────────────────────────────────────────────────────
def lhs_sample(n, bounds, seed=42):
    """Latin Hypercube Sample with n points in d dimensions, scaled to bounds."""
    from scipy.stats.qmc import LatinHypercube
    d = len(bounds)
    sampler = LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n)
    scaled = np.empty_like(sample)
    for i, (lo, hi) in enumerate(bounds):
        scaled[:, i] = lo + sample[:, i] * (hi - lo)
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
#  Single solve
# ─────────────────────────────────────────────────────────────────────────────
def _suppress_stdout():
    """Context manager to suppress stdout."""
    import io, contextlib
    return contextlib.redirect_stdout(io.StringIO())


def solve_one(params, verbose=False):
    """
    params: array of 8 values [R0, a, kappa, delta, Ip, betap, alpha_m, alpha_n]
    Returns dict or None on failure.
    """
    R0, a, kappa, delta, Ip, betap, alpha_m, alpha_n = params

    # Adaptive domain: 1.3× the plasma size with min margin 0.15 m
    margin = max(0.15, 0.3 * a)
    Rmin = R0 - a - margin
    Rmax = R0 + a + margin
    Zmax = kappa * a + margin
    Zmin = -Zmax

    try:
        eq = FixedBoundaryEquilibrium(
            R0=R0, a=a, kappa=kappa, delta=delta,
            Rmin=Rmin, Rmax=Rmax, Zmin=Zmin, Zmax=Zmax,
            nx=NX, ny=NY, order=ORDER, method=METHOD)

        pro = ConstrainBetapIp(
            betap=betap, Ip=Ip, fvac=1.0,
            alpha_m=alpha_m, alpha_n=alpha_n, Raxis=R0)

        with warnings.catch_warnings(), _suppress_stdout():
            warnings.simplefilter("ignore")
            errs = picard.solve(eq, pro, constrain=None,
                                maxits=MAXITS, rtol=RTOL,
                                anderson_m=ANDERSON_M,
                                convergenceInfo=True, verbose=False)

        converged = errs[-1][-1] < RTOL if errs else False
        niter = len(errs[0]) if errs else 0

        return {
            "psi":        np.asarray(eq.psi(), dtype=np.float32),
            "psi_axis":   float(eq.psi_axis),
            "psi_bndry":  float(eq.psi_bndry),
            "R_grid":     eq.R[:, 0].astype(np.float32),
            "Z_grid":     eq.Z[0, :].astype(np.float32),
            "converged":  converged,
            "niter":      niter,
            "Ip_actual":  float(eq.plasmaCurrent()),
            "betap_actual": float(eq.poloidalBeta()),
        }

    except Exception as e:
        if verbose:
            print(f"    ✗  {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  HDF5 I/O
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset(path, params_list, results_list):
    """Write dataset to HDF5."""
    import h5py
    N = len(params_list)
    # Determine max grid size (samples may have different domains)
    nx_list = [r["R_grid"].shape[0] for r in results_list]
    ny_list = [r["Z_grid"].shape[0] for r in results_list]
    nx_max, ny_max = max(nx_list), max(ny_list)

    with h5py.File(path, "w") as f:
        # Parameters
        f.create_dataset("parameters", data=np.array(params_list, dtype=np.float32))
        f.create_dataset("parameter_names", data=np.array(PARAM_NAMES, dtype="S"))

        # Converged / niter
        f.create_dataset("converged", data=np.array([r["converged"] for r in results_list], dtype=bool))
        f.create_dataset("niter", data=np.array([r["niter"] for r in results_list], dtype=np.int32))
        f.create_dataset("psi_axis", data=np.array([r["psi_axis"] for r in results_list], dtype=np.float32))
        f.create_dataset("psi_bndry", data=np.array([r["psi_bndry"] for r in results_list], dtype=np.float32))
        f.create_dataset("Ip_actual", data=np.array([r.get("Ip_actual", 0) for r in results_list], dtype=np.float32))
        f.create_dataset("betap_actual", data=np.array([r.get("betap_actual", 0) for r in results_list], dtype=np.float32))

        # Grids and ψ: variable-length via vlen or stored per-sample
        # Use compound approach: store each grid size + psi in separate datasets
        f.create_dataset("nx", data=np.array(nx_list, dtype=np.int32))
        f.create_dataset("ny", data=np.array(ny_list, dtype=np.int32))

        # Use a group with per-sample datasets for variable sizes
        grp = f.create_group("samples")
        for i in range(N):
            sgrp = grp.create_group(f"{i:06d}")
            sgrp.create_dataset("R_grid", data=results_list[i]["R_grid"])
            sgrp.create_dataset("Z_grid", data=results_list[i]["Z_grid"])
            sgrp.create_dataset("psi", data=results_list[i]["psi"])

        # Attributes
        f.attrs["description"] = "gspack v2.0 fixed-boundary GS dataset"
        f.attrs["n_samples"] = N
        f.attrs["param_bounds"] = str(PARAM_BOUNDS)
        f.attrs["nx"] = NX
        f.attrs["ny"] = NY
        f.attrs["maxits"] = MAXITS
        f.attrs["rtol"] = RTOL
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")


def load_dataset(path):
    """Quick-check loader — prints summary."""
    import h5py
    with h5py.File(path, "r") as f:
        params = f["parameters"][:]
        conv   = f["converged"][:]
        niter  = f["niter"][:]
        ntotal = f.attrs["n_samples"]
        nconv  = int(conv.sum())
    print(f"  Samples : {ntotal}  (converged: {nconv}, failed: {ntotal - nconv})")
    print(f"  Params  : {params.shape}")
    print(f"  Range   : [{params.min(axis=0)}, {params.max(axis=0)}]")
    print(f"  Mean iters: {niter[conv].mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate GS equilibrium dataset")
    parser.add_argument("--nsamples", type=int, default=100,
                        help="Number of samples (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HDF5 path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LHS")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from existing HDF5 checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint file path (default: auto)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample info")
    parser.add_argument("--quick", action="store_true",
                        help="Quick check — load and print existing dataset")
    args = parser.parse_args()

    bk.set_backend("cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "datasets")
    os.makedirs(out_dir, exist_ok=True)

    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(out_dir, f"gs_dataset_{stamp}.h5")
    if args.checkpoint is None:
        args.checkpoint = args.output.replace(".h5", "_checkpoint.npz")

    # Quick check mode
    if args.quick:
        if not os.path.exists(args.output):
            print(f"File not found: {args.output}")
            return
        load_dataset(args.output)
        return

    # Resume from checkpoint
    params_list  = []
    results_list = []
    start_idx = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume} ...")
        import h5py
        with h5py.File(args.resume, "r") as f:
            N0 = f.attrs["n_samples"]
            p0 = f["parameters"][:]
            conv0 = f["converged"][:]
            for i in range(N0):
                params_list.append(p0[i])
                sgrp = f[f"samples/{i:06d}"]
                results_list.append({
                    "psi":       sgrp["psi"][:],
                    "psi_axis":  float(f["psi_axis"][i]),
                    "psi_bndry": float(f["psi_bndry"][i]),
                    "R_grid":    sgrp["R_grid"][:],
                    "Z_grid":    sgrp["Z_grid"][:],
                    "converged": bool(conv0[i]),
                    "niter":     int(f["niter"][i]),
                })
        start_idx = len(params_list)
        print(f"  Loaded {start_idx} existing samples. Continuing ...")

    # Generate new samples
    if start_idx < args.nsamples:
        n_new = args.nsamples - start_idx
        bounds = [PARAM_BOUNDS[name] for name in PARAM_NAMES]
        samples = lhs_sample(n_new, bounds, seed=args.seed + start_idx)

        print("=" * 64)
        print(f"  Generating {n_new} new GS equilibria")
        print("=" * 64)
        print(f"  Grid : {NX}×{NY}")
        print(f"  Bounds:")
        for name, (lo, hi) in zip(PARAM_NAMES, bounds):
            print(f"    {name:8s}  [{lo:6.2g}, {hi:6.2g}]")
        print(f"  Output: {args.output}")
        print()

        # Progress bar
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_new), desc="Solving")
        except ImportError:
            iterator = range(n_new)

        n_conv = 0
        for idx in iterator:
            i = start_idx + idx
            params = samples[idx]
            result = solve_one(params, verbose=args.verbose)

            if result is not None:
                params_list.append(params)
                results_list.append(result)
                if result["converged"]:
                    n_conv += 1

            # Periodic checkpoint
            if (idx + 1) % max(1, n_new // 10) == 0 and (idx + 1) < n_new:
                save_dataset(args.checkpoint, params_list, results_list)
                if isinstance(iterator, tqdm):
                    iterator.set_postfix(conv=f"{n_conv}/{len(params_list)}")

        # Final save
        print(f"\n  Converged: {n_conv} / {len(params_list)}")
        print(f"  Saving to {args.output} ...")
        save_dataset(args.output, params_list, results_list)
        print("  Done.")

        # Clean up checkpoint
        if os.path.exists(args.checkpoint):
            os.remove(args.checkpoint)
    else:
        print(f"All {args.nsamples} samples already exist.")

    # Summary
    load_dataset(args.output)


if __name__ == "__main__":
    main()
