"""
Exponential distribution.

See Also
--------
scipy.stats.expon: Scipy equivalent.
"""

from math import expm1 as _expm1
from math import log1p as _log1p
from typing import Optional

import numpy as np

from ._util import (
    _generate_wrappers,
    _jit,
    _jit_pointwise,
    _prange,
    _rvs_jit,
    _seed,
    _trans,
)

_doc_par = """
loc : float
    Location of the mode.
scale : float
    Standard deviation.
"""


@_jit_pointwise(1)
def _cdf1(z: float) -> float:
    T = type(z)
    return T(0) if z < 0 else -_expm1(-z)


@_jit_pointwise(1)
def _ppf1(p: float) -> float:
    return -_log1p(-p)


@_jit(2)
def _logpdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    r = np.empty_like(z)
    for i in _prange(len(r)):
        r[i] = -np.inf if z[i] < 0 else -z[i] - np.log(scale)
    return r


@_jit(2)
def _pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, loc, scale))


@_jit(2)
def _cdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _cdf1(z[i])
    return z


@_jit(2)
def _ppf(p: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = np.empty_like(p)
    for i in _prange(len(z)):
        z[i] = _ppf1(p[i])
    return scale * z + loc


@_rvs_jit(2)
def _rvs(
    loc: float, scale: float, size: int, random_state: Optional[int]
) -> np.ndarray:
    _seed(random_state)
    return loc + np.random.exponential(scale, size)


_generate_wrappers(globals())
