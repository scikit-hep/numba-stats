"""
Crystal Ball distribution.

The Crystal Ball distribution replaces the lower tail of a normal distribution with
a power-law tail.

https://en.wikipedia.org/wiki/Crystal_Ball_function

See Also
--------
scipy.stats.crystalball: Scipy equivalent.
"""

from math import erf as _erf

import numpy as np

from . import norm as _norm
from ._util import _generate_wrappers, _jit, _jit_pointwise, _prange, _trans

_doc_par = """
x : Array-like
    Random variate.
beta : float
    Distance from the mode in units of standard deviations where the Crystal
    Ball turns from a gaussian into a power law.
m : float
    Absolute value of the slope of the powerlaw tail. Must be large than 1.
"""


@_jit_pointwise(3)
def _log_powerlaw(z: float, beta: float, m: float) -> float:
    T = type(beta)
    c = -T(0.5) * beta * beta
    log_a = m * np.log(m / beta) + c
    b = m / beta - beta
    return log_a - m * np.log(b - z)  # type:ignore[no-any-return]


@_jit_pointwise(3)
def _powerlaw_integral(z: float, beta: float, m: float) -> float:
    T = type(beta)
    log_a = m * np.log(m / beta) - T(0.5) * beta * beta
    b = m / beta - beta
    m1 = m - T(1)
    return np.exp(log_a - m1 * np.log(b - z) - np.log(m1))  # type:ignore[no-any-return]


@_jit_pointwise(2)
def _normal_integral(a: float, b: float) -> float:
    T = type(a)
    sqrt_half = np.sqrt(T(0.5))
    return sqrt_half * np.sqrt(T(np.pi)) * (_erf(b * sqrt_half) - _erf(a * sqrt_half))  # type:ignore[no-any-return]


@_jit_pointwise(3)
def _powerlaw_ppf(p: float, beta: float, m: float) -> float:
    T = type(beta)
    log_a = m * np.log(m / beta) - T(0.5) * beta * beta
    b = m / beta - beta
    m1 = m - T(1)
    log_term = (-T(1) / m1) * (np.log(p) + np.log(m1) - log_a)
    term = np.exp(log_term)
    return b - term  # type:ignore[no-any-return]


@_jit_pointwise(2, cache=False)
def _normal_ppf(p: float, a: float) -> float:
    # assumption is that p is always <= 1, so this function
    # never return NaN; caller is responsible for ensuring this
    T = type(a)
    sqrt_2pi = np.sqrt(T(2 * np.pi))
    cdf_a = _norm._cdf1(a)
    cdf_target = cdf_a + p / sqrt_2pi
    # protect against numerical rounding that can raise cdf_target above 1
    return _norm._ppf1(min(T(1), cdf_target))


@_jit_pointwise(3)
def _log_density(z: float, beta: float, m: float) -> float:
    if z < -beta:
        return _log_powerlaw(z, beta, m)
    return -0.5 * z * z


@_jit(4)
def _logpdf(
    x: np.ndarray, beta: float, m: float, loc: float, scale: float
) -> np.ndarray:
    z = _trans(x, loc, scale)
    norm = scale * (
        _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, type(beta)(np.inf))
    )
    c = np.log(norm)
    for i in _prange(len(z)):
        z[i] = _log_density(z[i], beta, m) - c
    return z


@_jit(4)
def _pdf(x: np.ndarray, beta: float, m: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, beta, m, loc, scale))


@_jit(4)
def _cdf(x: np.ndarray, beta: float, m: float, loc: float, scale: float) -> np.ndarray:
    T = type(beta)
    z = _trans(x, loc, scale)
    norm = _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, T(np.inf))
    for i in _prange(len(z)):
        if z[i] < -beta:
            z[i] = _powerlaw_integral(z[i], beta, m) / norm
        else:
            z[i] = (
                _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, z[i])
            ) / norm
    return z


@_jit(4, cache=False)
def _ppf(p: np.ndarray, beta: float, m: float, loc: float, scale: float) -> np.ndarray:
    T = type(beta)
    norm = _powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, T(np.inf))
    pbeta = _powerlaw_integral(-beta, beta, m) / norm
    r = np.empty_like(p)
    for i in _prange(len(r)):
        if p[i] < pbeta:
            r[i] = _powerlaw_ppf(p[i] * norm, beta, m)
        elif p[i] <= 1:
            r[i] = _normal_ppf((p[i] - pbeta) * norm, -beta)
        else:
            r[i] = np.nan
    return scale * r + loc


_generate_wrappers(globals())
