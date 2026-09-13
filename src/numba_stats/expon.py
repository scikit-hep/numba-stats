"""
Exponential distribution.

See Also
--------
scipy.stats.expon: Scipy equivalent.
"""

from math import expm1 as _expm1
from math import log1p as _log1p

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
def _rvs(loc: float, scale: float, size: int, random_state: int | None) -> np.ndarray:
    _seed(random_state)
    return loc + np.random.exponential(scale, size)


@_jit_pointwise(2)
def _integrate1(za: float, zb: float) -> float:
    # exp(-za) - exp(-zb) without cancellation
    T = type(za)
    za = max(za, T(0))
    zb = max(zb, T(0))
    return -np.exp(-za) * T(_expm1(za - zb))  # type:ignore[no-any-return]


@_jit_pointwise(4)
def _integrate(lo: float, hi: float, loc: float, scale: float) -> float:
    inv_scale = type(scale)(1) / scale
    return _integrate1((lo - loc) * inv_scale, (hi - loc) * inv_scale)


_generate_wrappers(globals())
