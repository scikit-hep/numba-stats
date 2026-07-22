"""
Exponentially modified normal distribution.

The distribution is defined as the distribution of the sum of a normal and 
exponential random variate. This definition has a right tail, but a left
tailed version can be defined by taking the difference between a normal and
exponential random variate instead.

To convert between the left- and right-tailed versions, a coordinate
transformation x -> 2\mu - x can be used to reflect the x-axis.

https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
See Also
--------
scipy.stats.exponnorm: Scipy equivalent.
"""

from math import erfc as _erfc, sqrt as _sqrt, exp as _exp, erf as _erf

import numpy as np

from . import norm as _norm
from ._util import _generate_wrappers, _jit, _jit_pointwise, _prange, _rvs_jit, _seed, _trans

_doc_par = """
loc : float
    Mode of unmodified normal distribution.
scale : float
    Standard deviation.
tau : float
    Exponential decay parameter.
"""

@_jit_pointwise(4)
def _pdf1(x: float, loc: float, scale: float, tau: float) -> float:
    T = type(x)
    sqrt2 = T(_sqrt(2))
    arg1 = (scale ** 2) / (T(2) * tau ** 2) - (x - loc) / tau
    arg2 = (scale) / (sqrt2 * tau) - (x - loc) / (sqrt2 * scale)
    return T(1) / (T(2) * tau) * _exp(arg1) * _erfc(arg2)

@_jit_pointwise(4)
def _cdf1(x: float, loc: float, scale: float, tau: float) -> float:
    T = type(x)
    sqrt2 = T(_sqrt(2))
    half = T(0.5)
    arg1 = (scale ** 2) / (T(2) * tau ** 2) - (x - loc) / tau
    z = _trans(x, loc, scale)
    arg2 = (scale) / (sqrt2 * tau) - (z / sqrt2)
    return _norm._cdf1(z) - half * _exp(arg1) * _erfc(arg2)

@_jit(3)
def _pdf(x: np.ndarray, loc: float, scale: float, tau: float) -> np.ndarray:
    r = np.empty_like(x)
    for i in _prange(len(x)):
        r[i] = _pdf1(x[i], loc, scale, tau)
    return r

@_jit(3)
def _cdf(x: np.ndarray , loc: float, scale: float, tau: float) -> np.ndarray :
    r = np.empty_like(x)
    for i in _prange(len(x)):
        r[i] = _cdf1(x[i], loc, scale, tau)
    return r

@_rvs_jit(3)
def _rvs(loc: float, scale: float, tau: float, size: int, random_state: int | None) -> np.ndarray:
    _seed(random_state)
    return np.random.normal(loc, scale, size) + np.random.exponential(tau, size)

_generate_wrappers(globals())