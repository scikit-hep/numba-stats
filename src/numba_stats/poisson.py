"""
Poisson distribution.

See Also
--------
scipy.stats.poisson: Scipy equivalent.
"""

from math import lgamma as _lgamma
from typing import Optional

import numba as nb
import numpy as np

from ._special import gammaincc as _gammaincc
from ._util import _generate_wrappers, _jit, _jit_custom, _prange, _seed

_doc_par = """
mu : float
    Expected value.
"""


@_jit(1)
def _logpmf(k: np.ndarray, mu: float) -> np.ndarray:
    T = type(mu)
    r = np.empty(len(k), T)
    for i in _prange(len(r)):
        if mu == 0:
            r[i] = 0.0 if k[i] == 0 else -np.inf
        else:
            r[i] = k[i] * np.log(mu) - _lgamma(k[i] + T(1)) - mu
    return r


@_jit(1)
def _pmf(k: np.ndarray, mu: float) -> np.ndarray:
    return np.exp(_logpmf(k, mu))


@_jit(1, cache=False)
def _cdf(k: np.ndarray, mu: float) -> np.ndarray:
    T = type(mu)
    r = np.empty(len(k), T)
    for i in _prange(len(r)):
        r[i] = _gammaincc(k[i] + T(1), mu)
    return r


@_jit_custom(nb.int64[:](nb.float32, nb.uint64, nb.optional(nb.uint64)))
def _rvs(mu: float, size: int, random_state: Optional[int]) -> np.ndarray:
    _seed(random_state)
    return np.random.poisson(mu, size)


_generate_wrappers(globals())
