"""
Normal distribution.

See Also
--------
scipy.stats.norm: Scipy equivalent.
"""

from math import erf as _erf
from math import erfc as _erfc

import numpy as np

from ._special import ndtri as _ndtri
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
    Location of the mode of the distribution.
scale : float
    Standard deviation.
"""


@_jit_pointwise(1)
def _logpdf1(z: float) -> float:
    T = type(z)
    return -T(0.5) * (z * z + T(np.log(2 * np.pi)))


@_jit_pointwise(1)
def _cdf1(z: float) -> float:
    T = type(z)
    c = T(np.sqrt(0.5))
    return T(0.5) * (T(1.0) + _erf(z * c))


@_jit_pointwise(1, cache=False)  # cannot cache because of _ndtri
def _ppf1(p: float) -> float:
    T = type(p)
    return T(_ndtri(p))


@_jit(2)
def _logpdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _logpdf1(r[i]) - np.log(scale)
    return r


@_jit(2)
def _pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, loc, scale))


@_jit(2)
def _cdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _cdf1(r[i])
    return r


@_jit(2, cache=False)
def _ppf(p: np.ndarray, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i]) + loc
    return r


@_rvs_jit(2)
def _rvs(loc: float, scale: float, size: int, random_state: int | None) -> np.ndarray:
    _seed(random_state)
    return np.random.normal(loc, scale, size)


@_jit_pointwise(2)
def _integrate1(za: float, zb: float) -> float:
    # integral of the standard normal density from za to zb; erfc is used on the
    # side where the interval lies to retain relative precision in the tails
    T = type(za)
    half = T(0.5)
    c = T(np.sqrt(0.5))
    if za + zb > 0:
        return half * (T(_erfc(za * c)) - T(_erfc(zb * c)))
    return half * (T(_erfc(-zb * c)) - T(_erfc(-za * c)))


@_jit_pointwise(4)
def _integrate(lo: float, hi: float, loc: float, scale: float) -> float:
    inv_scale = type(scale)(1) / scale
    return _integrate1((lo - loc) * inv_scale, (hi - loc) * inv_scale)


_generate_wrappers(globals())
