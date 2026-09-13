"""
Chi-squared distribution.

See Also
--------
scipy.stats.chi2: Scipy equivalent.
"""

from math import lgamma as _lgamma

import numpy as np

from ._special import gammainc as _gammainc
from ._special import gammaincc as _gammaincc
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
df : float
    Degrees of freedom. Must be positive.
loc : float
    Shift of the distribution.
scale : float
    Width parameter.
"""


@_jit_pointwise(2, cache=False)  # cannot cache because of _xlogy
def _logpdf1(z: float, df: float) -> float:
    T = type(z)
    if z < 0:
        return -T(np.inf)
    half = T(0.5)
    k = half * df
    return T(_xlogy(k - T(1), z)) - half * z - T(_lgamma(k)) - k * T(np.log(2))


@_jit_pointwise(2, cache=False)  # cannot cache because of _gammainc
def _cdf1(z: float, df: float) -> float:
    T = type(z)
    if z <= 0:
        return T(0)
    half = T(0.5)
    return T(_gammainc(half * df, half * z))


@_jit_pointwise(2, cache=False)  # cannot cache because of _gammaincinv
def _ppf1(p: float, df: float) -> float:
    T = type(p)
    if p == 0:
        return T(0)
    if p == 1:
        return T(np.inf)
    return T(2) * T(_gammaincinv(T(0.5) * df, p))


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    c = np.log(scale)
    for i in _prange(len(z)):
        z[i] = _logpdf1(z[i], df) - c
    return z


@_jit(3, cache=False)
def _pdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, df, loc, scale))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _cdf1(z[i], df)
    return z


@_jit(3, cache=False)
def _ppf(p: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i], df) + loc
    return r


@_rvs_jit(3)
def _rvs(
    df: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    return loc + scale * np.random.chisquare(df, size)


@_jit_pointwise(5, cache=False)
def _integrate(lo: float, hi: float, df: float, loc: float, scale: float) -> float:
    T = type(df)
    half = T(0.5)
    inv_scale = T(1) / scale
    za = max((lo - loc) * inv_scale, T(0))
    zb = max((hi - loc) * inv_scale, T(0))
    k = half * df
    if za + zb > T(2) * df:
        # interval beyond the mean, use the upper incomplete gamma function
        return T(_gammaincc(k, half * za)) - T(_gammaincc(k, half * zb))
    return T(_gammainc(k, half * zb)) - T(_gammainc(k, half * za))


_generate_wrappers(globals())
