"""
Laplace distribution.

See Also
--------
scipy.stats.laplace: Scipy equivalent.
"""

from math import expm1 as _expm1

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
    return 1.0 - 0.5 * np.exp(-z) if z > 0 else 0.5 * np.exp(z)  # type:ignore[no-any-return]


@_jit_pointwise(1)
def _ppf1(p: float) -> float:
    return -np.log(2 * (1 - p)) if p > 0.5 else np.log(2 * p)  # type:ignore[no-any-return]


@_jit(2)
def _logpdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    r = np.empty_like(z)
    for i in _prange(len(r)):
        r[i] = np.log(0.25) - np.abs(z[i])
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
    return np.random.laplace(loc, scale, size)


@_jit_pointwise(2)
def _integrate1(za: float, zb: float) -> float:
    T = type(za)
    half = T(0.5)
    if za >= 0 and zb >= 0:
        return -half * np.exp(-za) * T(_expm1(za - zb))  # type:ignore[no-any-return]
    if za <= 0 and zb <= 0:
        return half * np.exp(za) * T(_expm1(zb - za))  # type:ignore[no-any-return]
    return _cdf1(zb) - _cdf1(za)


@_jit_pointwise(4)
def _integrate(lo: float, hi: float, loc: float, scale: float) -> float:
    inv_scale = type(scale)(1) / scale
    return _integrate1((lo - loc) * inv_scale, (hi - loc) * inv_scale)


_generate_wrappers(globals())
