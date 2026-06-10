"""
gspack.picard  — Anderson-accelerated Picard iteration
"""
import numpy as np


class AndersonMixer:
    def __init__(self, m=5):
        self.m = m
        self._psi_hist = []
        self._res_hist = []

    def reset(self):
        self._psi_hist.clear();  self._res_hist.clear()

    def mix(self, psi_new, psi_old):
        if self.m == 0:
            return psi_new
        shape = psi_old.shape
        r_cur = (psi_new - psi_old).ravel()
        self._psi_hist.append(psi_old.ravel().copy())
        self._res_hist.append(r_cur.copy())
        if len(self._psi_hist) > self.m:
            self._psi_hist.pop(0);  self._res_hist.pop(0)
        m_cur = len(self._res_hist)
        if m_cur < 2:
            return psi_new
        F   = np.column_stack(self._res_hist)
        dF  = np.diff(F, axis=1)
        rhs = -F[:, -1]
        try:
            gamma, _, _, _ = np.linalg.lstsq(dF, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return psi_new
        c = np.zeros(m_cur)
        for i in range(m_cur - 1):
            c[i] -= gamma[i]
        c[-1] = 1.0 - c[:-1].sum()
        if abs(c.sum()) > 1e-10:
            c /= c.sum()
        psi_mixed = sum(c[k] * (self._psi_hist[k] + self._res_hist[k])
                        for k in range(m_cur))
        return psi_mixed.reshape(shape)


def solve(eq, profiles, constrain=None,
          maxits=50, rtol=1e-3, atol=1e-10,
          anderson_m=5,
          convergenceInfo=False, verbose=True):
    """
    Picard iteration with Anderson mixing acceleration.

    Parameters
    ----------
    eq, profiles, constrain : standard gspack objects
    maxits      : max iterations
    rtol        : relative psi change tolerance
    atol        : absolute psi change tolerance
    anderson_m  : Anderson window (0 = plain Picard)
    convergenceInfo / verbose : print table

    Returns (max_change_arr, rel_change_arr) when convergenceInfo=True.
    """
    show  = convergenceInfo or verbose
    mixer = AndersonMixer(m=anderson_m)

    if constrain is not None:
        constrain(eq)

    from .backend import to_numpy
    psi = to_numpy(eq.psi())

    max_list, rel_list = [], []

    if show:
        tag = f"Anderson m={anderson_m}" if anderson_m > 0 else "plain Picard"
        print(f"\n  {'Iter':>4}  {'max|Δψ|':>12}  {'rel|Δψ|':>12}  [{tag}]")
        print("  " + "-" * 52)

    for it in range(maxits):
        psi_last = psi.copy()

        eq.solve(profiles, psi=psi)
        psi_raw = to_numpy(eq.psi())
        psi     = mixer.mix(psi_raw, psi_last)

        delta      = np.abs(psi - psi_last)
        max_change = float(delta.max())
        span       = float(psi.max() - psi.min())
        rel_change = max_change / (span + 1e-30)

        max_list.append(max_change)
        rel_list.append(rel_change)

        if show:
            print(f"  {it+1:>4}  {max_change:>12.4e}  {rel_change:>12.4e}")

        if max_change < atol or rel_change < rtol:
            if show:
                print(f"  Converged at iteration {it+1}")
            break

        if constrain is not None:
            constrain(eq)
        psi = to_numpy(eq.psi())
    else:
        if show:
            print(f"  Warning: did not converge in {maxits} iterations")

    if convergenceInfo or verbose:
        return (np.array(max_list), np.array(rel_list))
