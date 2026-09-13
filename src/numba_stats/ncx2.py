"""
Noncentral chi-squared distribution.

See Also
--------
scipy.stats.ncx2: Scipy equivalent.
"""

import numpy as np

from . import chi2 as _chi2
from ._special import chndtr as _chndtr
from ._special import chndtrix as _chndtrix
from ._special import ive as _ive
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
nc : float
    Noncentrality parameter. Must be non-negative. For nc = 0, the distribution is the
    chi-squared distribution.
loc : float
    Shift of the distribution.
scale : float
    Width parameter.
"""


@_jit_pointwise(3, cache=False)  # cannot cache because of _ive
def _logpdf1(z: float, df: float, nc: float) -> float:
    # implementation taken from scipy.stats.ncx2, the factor exp(-zs * ns) is
    # absorbed into the exponentially scaled Bessel function for numerical stability
    if nc == 0:
        return _chi2._logpdf1(z, df)
    T = type(z)
    if z < 0:
        return -T(np.inf)
    half = T(0.5)
    df2 = half * df - T(1)
    zs = np.sqrt(z)
    ns = np.sqrt(nc)
    res = T(_xlogy(half * df2, z / nc)) - half * (zs - ns) ** 2
    corr = half * T(_ive(df2, zs * ns))
    if corr > 0:
        return res + np.log(corr)  # type:ignore[no-any-return]
    return -T(np.inf)


@_jit_pointwise(3, cache=False)  # cannot cache because of _chndtr
def _cdf1(z: float, df: float, nc: float) -> float:
    T = type(z)
    if z <= 0:
        return T(0)
    return T(_chndtr(z, df, nc))


@_jit_pointwise(3, cache=False)  # cannot cache because of _chndtrix
def _ppf1(p: float, df: float, nc: float) -> float:
    T = type(p)
    if p == 0:
        return T(0)
    if p == 1:
        return T(np.inf)
    return T(_chndtrix(p, df, nc))


@_jit(4, cache=False)
def _logpdf(
    x: np.ndarray, df: float, nc: float, loc: float, scale: float
) -> np.ndarray:
    z = _trans(x, loc, scale)
    c = np.log(scale)
    for i in _prange(len(z)):
        z[i] = _logpdf1(z[i], df, nc) - c
    return z


@_jit(4, cache=False)
def _pdf(x: np.ndarray, df: float, nc: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, df, nc, loc, scale))


@_jit(4, cache=False)
def _cdf(x: np.ndarray, df: float, nc: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _cdf1(z[i], df, nc)
    return z


@_jit(4, cache=False)
def _ppf(p: np.ndarray, df: float, nc: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i], df, nc) + loc
    return r


@_rvs_jit(4)
def _rvs(
    df: float, nc: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    return loc + scale * np.random.noncentral_chisquare(df, nc, size)


@_jit_pointwise(6, cache=False)
def _integrate(
    lo: float, hi: float, df: float, nc: float, loc: float, scale: float
) -> float:
    inv_scale = type(scale)(1) / scale
    return _cdf1((hi - loc) * inv_scale, df, nc) - _cdf1((lo - loc) * inv_scale, df, nc)


_generate_wrappers(globals())
