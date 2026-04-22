"""
Binomial distribution.

See Also
--------
scipy.stats.binom: Scipy equivalent.
"""

from math import lgamma as _lgamma

import numba as nb
import numpy as np

from ._special import betainc as _betainc
from ._special import xlog1py as _xlog1py
from ._special import xlogy as _xlogy
from ._util import _generate_wrappers, _jit, _prange, _seed

_doc_par = """
k : int
    Number of successes.
n : int
    Number of trials.
p : float
    Success probability for each trial.
"""


@_jit(1, narg=2, cache=False)
def _logpmf(k: np.ndarray, n: np.ndarray, p: float) -> np.ndarray:
    T = type(p)
    r = np.empty(len(k), T)
    one = T(1)
    for i in _prange(len(r)):
        combiln = _lgamma(n[i] + one) - (
            _lgamma(k[i] + one) + _lgamma(n[i] - k[i] + one)
        )
        r[i] = combiln + _xlogy(k[i], p) + _xlog1py(n[i] - k[i], -p)
    return r


@_jit(1, narg=2, cache=False)
def _pmf(k: np.ndarray, n: np.ndarray, p: float) -> np.ndarray:
    return np.exp(_logpmf(k, n, p))


@_jit(1, narg=2, cache=False)
def _cdf(k: np.ndarray, n: np.ndarray, p: float) -> np.ndarray:
    T = type(p)
    r = np.empty(len(k), T)
    one = T(1)
    for i in _prange(len(r)):
        if k[i] == n[i]:
            r[i] = 1
        elif p == 0:
            r[i] = 1
        elif p == 1:
            r[i] = 0
        else:
            r[i] = 1 - _betainc(k[i] + one, n[i] - k[i], p)
    return r


@nb.njit(  # type:ignore[misc]
    nb.int64[:](nb.uint64, nb.float32, nb.uint64, nb.optional(nb.uint64)),
    cache=True,
    inline="always",
    error_model="numpy",
)
def _rvs(n: int, p: float, size: int, random_state: int | None) -> np.ndarray:
    _seed(random_state)
    return np.random.binomial(n, p, size=size)


_generate_wrappers(globals())
