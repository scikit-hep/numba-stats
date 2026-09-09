"""
Gamma distribution.

See Also
--------
scipy.stats.gamma: Scipy equivalent.
"""

from math import lgamma as _lgamma

import numpy as np

from ._special import gammainc as _gammainc
from ._special import gammaincinv as _gammaincinv
from ._special import xlogy as _xlogy
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
a : float
    Shape parameter. Must be positive.
loc : float
    Shift of the distribution.
scale : float
    Width parameter.
"""


@_jit_pointwise(2, cache=False)  # cannot cache because of _xlogy
def _logpdf1(z: float, a: float) -> float:
    T = type(z)
    if z < 0:
        return -T(np.inf)
    return T(_xlogy(a - T(1), z)) - z - T(_lgamma(a))


@_jit_pointwise(2, cache=False)  # cannot cache because of _gammainc
def _cdf1(z: float, a: float) -> float:
    T = type(z)
    if z <= 0:
        return T(0)
    return T(_gammainc(a, z))


@_jit_pointwise(2, cache=False)  # cannot cache because of _gammaincinv
def _ppf1(p: float, a: float) -> float:
    T = type(p)
    return T(_gammaincinv(a, p))


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    c = np.log(scale)
    for i in _prange(len(z)):
        z[i] = _logpdf1(z[i], a) - c
    return z


@_jit(3, cache=False)
def _pdf(x: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, a, loc, scale))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _cdf1(z[i], a)
    return z


@_jit(3, cache=False)
def _ppf(p: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i], a) + loc
    return r


@_rvs_jit(3)
def _rvs(
    a: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    return loc + np.random.gamma(a, scale, size)


_generate_wrappers(globals())
