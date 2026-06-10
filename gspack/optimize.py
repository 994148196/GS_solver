"""
gspack.optimize
===============
Plasma shape optimisation and uncertainty quantification.

Functions
---------
optimize_shape    — find coil currents that achieve target κ, δ, R
monte_carlo_uq    — Monte Carlo UQ for coil current / position errors
"""

import numpy as np
import warnings


# ─────────────────────────────────────────────────────────────────────────────
#  Shape optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimize_shape(eq, profiles, constrain_obj=None,
                   target_kappa=1.6,
                   target_delta=0.33,
                   target_R_mag=1.25,
                   target_Ip=None,
                   weights=(1.0, 1.0, 1.0),
                   method='Nelder-Mead',
                   maxits_picard=10,
                   maxits_opt=200,
                   verbose=False):
    """
    Optimise coil currents to achieve target plasma shape.

    Parameters
    ----------
    eq              : Equilibrium (will be modified in place)
    profiles        : Profile object
    constrain_obj   : control.constrain (or None — shape only)
    target_kappa    : target elongation κ
    target_delta    : target triangularity δ
    target_R_mag    : target magnetic axis R [m]
    target_Ip       : target plasma current [A] (if None, not constrained)
    weights         : (w_kappa, w_delta, w_R[, w_Ip]) — objective weights
    method          : scipy.optimize.minimize method
    maxits_picard   : inner Picard iterations per evaluation
    maxits_opt      : max optimiser iterations
    verbose         : print objective value at each step

    Returns
    -------
    result : scipy.optimize.OptimizeResult
    """
    from scipy.optimize import minimize
    from . import picard as _picard

    tok = eq.tokamak
    I0  = np.array(tok.controlCurrents(), dtype=float)
    w   = list(weights)
    if len(w) < 4:
        w = list(w) + [0.0]

    call_count = [0]

    def objective(I_coils):
        call_count[0] += 1
        tok.setControlCurrents(I_coils)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _picard.solve(eq, profiles, constrain_obj,
                              maxits=maxits_picard,
                              convergenceInfo=False)
        except Exception:
            return 1e12

        loss  = w[0] * (eq.elongation()    - target_kappa)**2
        loss += w[1] * (eq.triangularity() - target_delta)**2
        loss += w[2] * (eq.magneticAxis()[0] - target_R_mag)**2
        if target_Ip is not None and w[3] > 0:
            loss += w[3] * ((eq.plasmaCurrent() - target_Ip) / (target_Ip + 1e-30))**2

        if verbose:
            print(f"  iter {call_count[0]:4d}  loss={loss:.4e}  "
                  f"κ={eq.elongation():.3f}  δ={eq.triangularity():.3f}  "
                  f"R={eq.magneticAxis()[0]:.4f}")
        return loss

    result = minimize(
        objective, I0,
        method=method,
        options={'xatol': 1.0, 'fatol': 1e-6, 'maxiter': maxits_opt,
                 'disp': False}
    )

    # Apply best currents and resolve
    tok.setControlCurrents(result.x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _picard.solve(eq, profiles, constrain_obj,
                      maxits=50, convergenceInfo=False)

    print(f"  Optimisation {'converged' if result.success else 'stopped'}  "
          f"after {call_count[0]} evaluations")
    print(f"  Final:  κ={eq.elongation():.4f}  δ={eq.triangularity():.4f}  "
          f"R_mag={eq.magneticAxis()[0]:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Monte-Carlo uncertainty quantification
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_uq(eq, profiles, constrain_obj,
                   n_samples=200,
                   sigma_I=100.0,
                   sigma_R_coil=0.001,
                   sigma_Z_coil=0.001,
                   maxits_picard=20,
                   random_seed=42,
                   verbose=True):
    """
    Monte-Carlo uncertainty quantification for coil errors.

    Perturbs coil currents and positions by Gaussian noise and
    reports statistics of key equilibrium quantities.

    Parameters
    ----------
    eq              : Equilibrium (nominal, used as starting point)
    profiles        : Profile object
    constrain_obj   : control.constrain
    n_samples       : number of Monte-Carlo samples
    sigma_I         : coil current 1-σ error [A]
    sigma_R_coil    : coil R position 1-σ error [m]
    sigma_Z_coil    : coil Z position 1-σ error [m]
    maxits_picard   : Picard iterations per sample
    random_seed     : numpy random seed
    verbose         : print progress

    Returns
    -------
    results : dict of lists  {'kappa', 'delta', 'psi_axis', 'R_mag', 'q95', 'Ip'}
    stats   : dict of summary statistics (mean, std, p5, p95)
    """
    from . import picard as _picard

    rng = np.random.default_rng(random_seed)
    tok = eq.tokamak

    nominal_I  = np.array(tok.controlCurrents(), dtype=float)
    nominal_Rc = np.array([c.R for _, c in tok.coils], dtype=float)
    nominal_Zc = np.array([c.Z for _, c in tok.coils], dtype=float)
    n_ctrl     = len(nominal_I)

    keys    = ['kappa', 'delta', 'psi_axis', 'R_mag', 'q95', 'Ip']
    results = {k: [] for k in keys}
    n_ok    = 0

    if verbose:
        print(f"  Monte-Carlo UQ: {n_samples} samples, "
              f"σ_I={sigma_I:.0f} A, σ_R={sigma_R_coil*1e3:.1f} mm")

    for i in range(n_samples):
        # Perturb
        delta_I = rng.normal(0.0, sigma_I, n_ctrl)
        tok.setControlCurrents(nominal_I + delta_I)
        for j, (_, coil) in enumerate(tok.coils):
            coil.R = nominal_Rc[j] + rng.normal(0.0, sigma_R_coil)
            coil.Z = nominal_Zc[j] + rng.normal(0.0, sigma_Z_coil)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _picard.solve(eq, profiles, constrain_obj,
                              maxits=maxits_picard, convergenceInfo=False)

            results['kappa'].append(eq.elongation())
            results['delta'].append(eq.triangularity())
            results['psi_axis'].append(eq.psi_axis)
            results['R_mag'].append(eq.magneticAxis()[0])
            q95 = float(eq.q(np.array([0.95]))[0]) if eq._opoints else float('nan')
            results['q95'].append(q95)
            results['Ip'].append(eq.plasmaCurrent())
            n_ok += 1

        except Exception:
            pass   # skip non-converged samples

        if verbose and (i+1) % max(1, n_samples//5) == 0:
            print(f"    {i+1}/{n_samples} samples  ({n_ok} converged)")

    # Restore nominal
    tok.setControlCurrents(nominal_I)
    for j, (_, coil) in enumerate(tok.coils):
        coil.R = nominal_Rc[j]
        coil.Z = nominal_Zc[j]

    # Summary statistics
    stats = {}
    for k, vals in results.items():
        if vals:
            arr = np.array([v for v in vals if np.isfinite(v)])
            stats[k] = {
                'mean': float(np.mean(arr)),
                'std':  float(np.std(arr)),
                'p5':   float(np.percentile(arr,  5)),
                'p95':  float(np.percentile(arr, 95)),
                'min':  float(arr.min()),
                'max':  float(arr.max()),
            }

    if verbose:
        print(f"\n  UQ summary ({n_ok}/{n_samples} converged):")
        print(f"  {'Quantity':<12}  {'mean':>10}  {'std':>10}  "
              f"{'p5':>10}  {'p95':>10}")
        print("  " + "-"*54)
        for k, s in stats.items():
            print(f"  {k:<12}  {s['mean']:>10.4g}  {s['std']:>10.4g}  "
                  f"  {s['p5']:>10.4g}  {s['p95']:>10.4g}")

    return results, stats
