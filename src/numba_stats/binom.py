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
from ._util import _generate_wrappers, _jit, _jit_pointwise, _prange, _seed

_doc_par = """
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


@_jit_pointwise(3, cache=False)  # cannot cache because of _betainc
def _cdf1(k: float, n: float, p: float) -> float:
    T = type(p)
    if k < 0:
        return T(0)
    if k >= n or p == 0:
        return T(1)
    if p == 1:
        return T(0)
    return T(1) - T(_betainc(k + T(1), n - k, p))


@_jit_pointwise(3, cache=False)  # cannot cache because of _betainc
def _sf1(k: float, n: float, p: float) -> float:
    # probability of more than k successes, 1 - cdf without cancellation
    T = type(p)
    if k < 0:
        return T(1)
    if k >= n or p == 0:
        return T(0)
    if p == 1:
        return T(1)
    return T(_betainc(k + T(1), n - k, p))


@_jit(1, narg=2, cache=False)
def _cdf(k: np.ndarray, n: np.ndarray, p: float) -> np.ndarray:
    r = np.empty(len(k), type(p))
    for i in _prange(len(r)):
        r[i] = _cdf1(k[i], n[i], p)
    return r


@_jit_pointwise(4, cache=False)
def _integrate(lo: float, hi: float, n: float, p: float) -> float:
    # probability of lo < k <= hi, equal to cdf(hi) - cdf(lo)
    if lo + hi > type(p)(2) * n * p:
        # interval beyond the mean
        return _sf1(lo, n, p) - _sf1(hi, n, p)
    return _cdf1(hi, n, p) - _cdf1(lo, n, p)


@nb.njit(  # type:ignore[untyped-decorator]
    nb.int64[:](nb.uint64, nb.float32, nb.uint64, nb.optional(nb.uint64)),
    cache=True,
    inline="always",
    error_model="numpy",
)
def _rvs(n: int, p: float, size: int, random_state: int | None) -> np.ndarray:
    _seed(random_state)
    return np.random.binomial(n, p, size=size)


_generate_wrappers(globals())
