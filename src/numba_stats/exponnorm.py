"""
Exponentially modified normal distribution.

The distribution of the sum of a normal and an exponential random variate. It has a
right tail. The left-tailed version, which is used to model signal loss in detectors,
for example, in fits of gamma lines, is obtained by reflecting the variate around loc,
by replacing x with 2 * loc - x.

https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution

Notes
-----
The shape parameter K is the mean of the exponential component divided by the standard
deviation of the normal component, K = tau / scale, which follows the convention of
scipy.stats.exponnorm.

See Also
--------
scipy.stats.exponnorm: Scipy equivalent.
"""

from math import erfc as _erfc

import numpy as np

from ._special import erfcx as _erfcx
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
K : float
    Shape parameter. Mean of the exponential component divided by the standard
    deviation of the normal component. Must be positive.
loc : float
    Location of the mode of the normal component.
scale : float
    Standard deviation of the normal component.
"""


@_jit_pointwise(2, cache=False)  # cannot cache because of _erfcx
def _logpdf1(z: float, K: float) -> float:
    # The textbook formula exp(v * v / 2 - z * v) * erfc(u) with v = 1 / K and
    # u = (v - z) / sqrt(2) overflows for small K. We use the identity
    # v * v / 2 - z * v - u * u = -z * z / 2 to rewrite it in terms of the scaled
    # complementary error function erfcx(u) = exp(u * u) * erfc(u) for u >= 0.
    T = type(z)
    half = T(0.5)
    v = T(1) / K
    u = (v - z) * T(np.sqrt(0.5))
    c = -np.log(T(2) * K)
    if u >= 0:
        return c - half * z * z + np.log(T(_erfcx(u)))  # type:ignore[no-any-return]
    # erfc(u) is between 1 and 2 here and the exponent is negative
    return c + half * v * v - z * v + np.log(T(_erfc(u)))  # type:ignore[no-any-return]


@_jit_pointwise(2, cache=False)  # cannot cache because of _erfcx
def _cdf1(z: float, K: float) -> float:
    # see _logpdf1 for the rationale of the two branches
    T = type(z)
    half = T(0.5)
    v = T(1) / K
    u = (v - z) * T(np.sqrt(0.5))
    if u >= 0:
        corr = np.exp(-half * z * z) * T(_erfcx(u))
    else:
        corr = np.exp(half * v * v - z * v) * T(_erfc(u))
    # erfc(-z) instead of 1 + erf(z) to retain relative precision in the left tail
    return half * (T(_erfc(-z * T(np.sqrt(0.5)))) - corr)  # type:ignore[no-any-return]


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, K: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    c = np.log(scale)
    for i in _prange(len(z)):
        z[i] = _logpdf1(z[i], K) - c
    return z


@_jit(3, cache=False)
def _pdf(x: np.ndarray, K: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, K, loc, scale))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, K: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _cdf1(z[i], K)
    return z


@_rvs_jit(3)
def _rvs(
    K: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    return np.random.normal(loc, scale, size) + np.random.exponential(K * scale, size)


_generate_wrappers(globals())
