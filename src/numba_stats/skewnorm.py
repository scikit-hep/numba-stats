"""
Skew-normal distribution.

A normal distribution with an additional shape parameter a, which introduces
skewness. The distribution is skewed to the right for a > 0 and to the left for a < 0.
For a = 0, it is a normal distribution.

https://en.wikipedia.org/wiki/Skew_normal_distribution

Notes
-----
The cdf is computed with Owen's T function, which limits the absolute precision to
about 1e-16. For a > 0, the relative precision of the cdf in the left tail is
therefore poor, where the cdf is smaller than about 1e-10, and likewise for 1 - cdf in
the right tail for a < 0.

See Also
--------
scipy.stats.skewnorm: Scipy equivalent.
"""

import numpy as np

from . import norm as _norm
from ._special import log_ndtr as _log_ndtr
from ._special import owens_t as _owens_t
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
    Skewness parameter. The distribution is skewed to the right for positive values
    and to the left for negative values. For a = 0, it is a normal distribution.
loc : float
    Location parameter.
scale : float
    Width parameter.
"""


@_jit_pointwise(2, cache=False)  # cannot cache because of _log_ndtr
def _logpdf1(z: float, a: float) -> float:
    T = type(z)
    return T(np.log(2)) + _norm._logpdf1(z) + T(_log_ndtr(a * z))


@_jit_pointwise(2, cache=False)  # cannot cache because of _owens_t
def _cdf1(z: float, a: float) -> float:
    T = type(z)
    return _norm._cdf1(z) - T(2) * T(_owens_t(z, a))


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    c = np.log(scale)
    for i in _prange(len(r)):
        r[i] = _logpdf1(r[i], a) - c
    return r


@_jit(3, cache=False)
def _pdf(x: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, a, loc, scale))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _cdf1(r[i], a)
    return r


@_rvs_jit(3)
def _rvs(
    a: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    # method used by scipy.stats.skewnorm, see
    # https://github.com/scipy/scipy/blob/v1.18.0/scipy/stats/_continuous_distns.py#L9724-L9946
    _seed(random_state)
    u0 = np.random.standard_normal(size)
    v = np.random.standard_normal(size)
    d = a / np.sqrt(1 + a * a)
    u1 = d * u0 + np.sqrt(1 - d * d) * v
    return loc + scale * np.where(u0 >= 0, u1, -u1)  # type:ignore[no-any-return]


_generate_wrappers(globals())
