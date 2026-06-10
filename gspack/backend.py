"""
gspack.backend
==============
CPU / GPU transparent array backend.

Design rule for other modules
------------------------------
  NEVER `from .backend import xp` at module level — stale after set_backend().
  ALWAYS call `get_xp()` inside functions that need the array module.
  Module-level `from .backend import to_numpy, to_backend, asarray, MU0` is fine.

Elliptic integrals on GPU
--------------------------
cupyx.scipy.special is optional; we fall back to computing on CPU + upload.
"""

import numpy as _np

MU0 = 4e-7 * _np.pi

_state = {'backend': 'cpu', 'xp': _np, 'cupy_ok': False}

try:
    import cupy as _cp
    _state['cupy_ok'] = True
except ImportError:
    _cp = None


def _init(mode='auto'):
    if mode == 'auto':
        mode = 'gpu' if _state['cupy_ok'] else 'cpu'
    if mode == 'gpu':
        if not _state['cupy_ok']:
            raise ImportError(
                "CuPy not found. Install: pip install cupy-cuda12x\n"
                "Or use set_backend('cpu').")
        _state['xp']      = _cp
        _state['backend'] = 'gpu'
    else:
        _state['xp']      = _np
        _state['backend'] = 'cpu'


def set_backend(mode='auto'):
    """Change backend at runtime.  mode: 'auto' | 'cpu' | 'gpu'"""
    _init(mode)
    print(f"[gspack] backend = {_state['backend']}")


def get_backend():
    return _state['backend']


def get_xp():
    """Return active array module (numpy or cupy)."""
    return _state['xp']


# ── Elliptic integrals ───────────────────────────────────────────────────────

def ellipk_compat(k2):
    """K(k²) — works on CPU and GPU arrays."""
    if _state['backend'] == 'gpu' and _state['cupy_ok']:
        try:
            from cupyx.scipy.special import ellipk as _gek
            return _gek(k2)
        except (ImportError, AttributeError):
            pass
        # CPU fallback with GPU upload
        from scipy.special import ellipk as _cek
        k2_np = _cp.asnumpy(k2) if isinstance(k2, _cp.ndarray) else _np.asarray(k2)
        return _cp.asarray(_cek(k2_np))
    from scipy.special import ellipk as _cek
    return _cek(k2)


def ellipe_compat(k2):
    """E(k²) — works on CPU and GPU arrays."""
    if _state['backend'] == 'gpu' and _state['cupy_ok']:
        try:
            from cupyx.scipy.special import ellipe as _gee
            return _gee(k2)
        except (ImportError, AttributeError):
            pass
        from scipy.special import ellipe as _cee
        k2_np = _cp.asnumpy(k2) if isinstance(k2, _cp.ndarray) else _np.asarray(k2)
        return _cp.asarray(_cee(k2_np))
    from scipy.special import ellipe as _cee
    return _cee(k2)


# ── Array conversion helpers ─────────────────────────────────────────────────

def to_numpy(arr):
    """Convert any array to plain NumPy ndarray."""
    if _state['backend'] == 'gpu' and _state['cupy_ok']:
        if isinstance(arr, _cp.ndarray):
            return _cp.asnumpy(arr)
    return _np.asarray(arr)


def to_backend(arr):
    """Move array (or scalar) to active backend device."""
    if arr is None:
        return None
    xp = _state['xp']
    if _state['backend'] == 'gpu' and _state['cupy_ok']:
        if isinstance(arr, _cp.ndarray):
            return arr
        # Convert to numpy first — avoids "Unsupported type numpy.ndarray" in CuPy
        return _cp.asarray(_np.asarray(arr, dtype=float))
    return _np.asarray(arr, dtype=float)


def asarray(x, dtype=float):
    """Convert scalar/array to ndarray on active device."""
    if _state['backend'] == 'gpu' and _state['cupy_ok']:
        if isinstance(x, _cp.ndarray):
            return x
        return _cp.asarray(_np.asarray(x, dtype=dtype))
    return _np.asarray(x, dtype=dtype)


_init('auto')
