"""
Skewed normal distribution.

https://en.wikipedia.org/wiki/Skew_normal_distribution

See Also
--------
scipy.stats.skewnorm: Scipy equivalent.
"""

from math import sqrt as _sqrt

import numpy as np

from . import norm as _norm
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
loc : float
    Location parameter.
scale : float
    Scale parameter.
a : float
    Skewness parameter.
"""


@_jit_pointwise(2)
def _logpdf1(z: float, a: float) -> float:
    T = type(z)
    return T(np.log(2)) + _norm._logpdf1(z) + T(np.log(_norm._cdf1(a * z)))


@_jit(3)
def _logpdf(x: np.ndarray, loc: float, scale: float, a: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _logpdf1(r[i], a) - np.log(scale)
    return r


@_jit(3)
def _pdf(x: np.ndarray, loc: float, scale: float, a: float) -> np.ndarray:
    return np.exp(_logpdf(x, loc, scale, a))


@_jit_pointwise(2, cache=False)
def _cdf1(z: float, a: float) -> float:
    T = type(z)
    return _norm._cdf1(z) - T(2) * T(_owens_t(z, a))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, loc: float, scale: float, a: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(x)):
        r[i] = _cdf1(r[i], a)
    return r


@_rvs_jit(3)
def _rvs(
    loc: float, scale: float, a: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    # Implementation taken from scipy
    # https://github.com/scipy/scipy/blob/v1.18.0/scipy/stats/_continuous_distns.py#L9724-L9946
    u0 = np.random.normal(loc=0, scale=1, size=size)
    v = np.random.normal(loc=0, scale=1, size=size)
    d = a / _sqrt(1 + a**2)
    u1 = d * u0 + v * _sqrt(1 - d**2)
    return loc + scale * np.where(u0 >= 0, u1, -u1)


_generate_wrappers(globals())
