"""
gspack v2.0 — 05-visualize-dataset
===================================
Randomly pick samples from an HDF5 dataset and plot the ψ(R,Z) distribution.

Usage:
  python examples/05_visualize_dataset.py                          # latest dataset
  python examples/05_visualize_dataset.py --input path/to/file.h5  # specific file
  python examples/05_visualize_dataset.py --nsamples 6             # 6 per page
  python examples/05_visualize_dataset.py --nrows 3 --ncols 4      # grid layout
"""

import sys, os, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _find_latest_dataset(datasets_dir):
    """Return the most recent .h5 file in datasets/."""
    files = sorted(glob.glob(os.path.join(datasets_dir, "gs_dataset_*.h5")))
    if not files:
        # fallback: look in examples/../
        files = sorted(glob.glob(os.path.join(datasets_dir, "..", "datasets", "gs_dataset_*.h5")))
    return files[-1] if files else None


def _param_str(params, param_names, values_extra=None):
    """Format parameters for title, in two lines."""
    lines = []
    # LCFS params
    items = [
        (r"$R_0$", params[0], ".2f"),
        (r"$a$",   params[1], ".2f"),
        (r"$\kappa$", params[2], ".2f"),
        (r"$\delta$", params[3], ".2f"),
    ]
    line1 = "  ".join(f"{name}={val:{fmt}}" for name, val, fmt in items)
    lines.append(line1)

    # Profile params
    items = [
        (r"$I_p$",     params[4], ".1e"),
        (r"$\beta_p$", params[5], ".2f"),
        (r"$\alpha_m$", params[6], ".2f"),
        (r"$\alpha_n$", params[7], ".2f"),
    ]
    line2 = "  ".join(f"{name}={val:{fmt}}" for name, val, fmt in items)
    lines.append(line2)

    # Optional extra values (psi_axis, psi_bndry, converged, niter)
    if values_extra is not None:
        extra = (f"ψₐ={values_extra.get('psi_axis', 0):.3f}  "
                 f"ψ_b={values_extra.get('psi_bndry', 0):.3f}  "
                 f"n={values_extra.get('niter', 0)}"
                 f"{'  ✓' if values_extra.get('converged') else '  ✗'}")
        lines.append(extra)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_samples(dataset_path, nsamples=6, nrows=2, ncols=3, output=None):
    """Pick `nsamples` random samples from dataset and plot ψ."""
    with h5py.File(dataset_path, "r") as f:
        N = f.attrs["n_samples"]
        param_names = [n.decode() for n in f["parameter_names"][:]]
        params_all = f["parameters"][:]
        psi_axis_all = f["psi_axis"][:]
        psi_bndry_all = f["psi_bndry"][:]
        converged_all = f["converged"][:]
        niter_all = f["niter"][:]

        # Random pick
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(N, size=min(nsamples, N), replace=False)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
        axes = np.atleast_1d(axes).ravel()

        for ax_idx, sample_idx in enumerate(indices):
            if ax_idx >= len(axes):
                break
            ax = axes[ax_idx]

            # Load ψ and grid
            sgrp = f[f"samples/{sample_idx:06d}"]
            psi = sgrp["psi"][:]
            R_grid = sgrp["R_grid"][:]
            Z_grid = sgrp["Z_grid"][:]
            R2d, Z2d = np.meshgrid(R_grid, Z_grid, indexing="ij")

            params = params_all[sample_idx]

            # Contourf
            levels = 24
            cf = ax.contourf(R2d, Z2d, psi, levels=levels, cmap="RdYlBu_r")
            ax.contour(R2d, Z2d, psi,
                       levels=np.linspace(psi_bndry_all[sample_idx],
                                          psi_axis_all[sample_idx], 10),
                       colors="white", linewidths=0.5)

            # Title
            extra = {
                "psi_axis": psi_axis_all[sample_idx],
                "psi_bndry": psi_bndry_all[sample_idx],
                "converged": converged_all[sample_idx],
                "niter": niter_all[sample_idx],
            }
            title = _param_str(params, param_names, extra)
            ax.set_title(title, fontsize=8, loc="left")
            ax.set_xlabel("R [m]", fontsize=7)
            ax.set_ylabel("Z [m]", fontsize=7)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)

        # Hide unused axes
        for ax_idx in range(len(indices), len(axes)):
            axes[ax_idx].set_visible(False)

        fig.suptitle(f"GS dataset — random samples from {os.path.basename(dataset_path)}",
                     fontsize=11)
        plt.tight_layout()

        if output is None:
            stamp = os.path.splitext(os.path.basename(dataset_path))[0]
            output = os.path.join(os.path.dirname(dataset_path), f"{stamp}_samples.png")
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualise GS dataset samples")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to HDF5 dataset (default: latest in datasets/)")
    parser.add_argument("--nsamples", type=int, default=6,
                        help="Number of samples to plot (default: 6)")
    parser.add_argument("--nrows", type=int, default=2,
                        help="Number of rows (default: 2)")
    parser.add_argument("--ncols", type=int, default=3,
                        help="Number of cols (default: 3)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path")
    args = parser.parse_args()

    # Locate dataset
    if args.input is not None:
        dataset_path = args.input
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, "..", "datasets")
        dataset_path = _find_latest_dataset(datasets_dir)

    if dataset_path is None or not os.path.exists(dataset_path):
        print("  No dataset found. Run 04_generate_dataset.py first.")
        return

    print(f"  Dataset: {dataset_path}")
    plot_samples(dataset_path,
                 nsamples=args.nsamples,
                 nrows=args.nrows,
                 ncols=args.ncols,
                 output=args.output)


if __name__ == "__main__":
    main()
