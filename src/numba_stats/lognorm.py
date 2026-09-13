"""
Lognormal distribution.

See Also
--------
scipy.stats.lognorm: Scipy equivalent.
"""

import numpy as np

from . import norm as _norm
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
s : float
    Standard deviation of the corresponding normal distribution of exp(x).
loc : float
    Shift of the distribution.
scale : float
    Equal to exp(mu) of the corresponding normal distribution of exp(x).
"""


@_jit(3)
def _logpdf(x: np.ndarray, s: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        if r[i] > 0:
            r[i] = -0.5 * np.log(r[i]) ** 2 / s**2 - np.log(
                s * r[i] * np.sqrt(2 * np.pi) * scale
            )
        else:
            r[i] = -np.inf
    return r


@_jit(3)
def _pdf(x: np.ndarray, s: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, s, loc, scale))


@_jit(3)
def _cdf(x: np.ndarray, s: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        if r[i] <= 0:
            r[i] = 0.0
        else:
            z = np.log(r[i]) / s
            r[i] = _norm._cdf1(z)
    return r


@_jit(3, cache=False)  # no cache because of norm._ppf
def _ppf(p: np.ndarray, s: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(p)):
        r[i] = np.exp(s * _norm._ppf1(p[i]))
    return scale * r + loc


@_rvs_jit(3, cache=False)
def _rvs(
    s: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    return loc + scale * np.random.lognormal(0, s, size)


@_jit_pointwise(5)
def _integrate(lo: float, hi: float, s: float, loc: float, scale: float) -> float:
    T = type(s)
    inv_scale = T(1) / scale
    za = (lo - loc) * inv_scale
    zb = (hi - loc) * inv_scale
    ua = np.log(za) / s if za > 0 else -T(np.inf)
    ub = np.log(zb) / s if zb > 0 else -T(np.inf)
    return _norm._integrate1(ua, ub)


_generate_wrappers(globals())
