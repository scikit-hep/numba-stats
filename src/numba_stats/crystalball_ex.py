"""
Generalised Crystal Ball distribution (aka double-sided crystal ball).

The generalised Crystal Ball distribution replaces the lower and upper tail of
an asymmetric normal distribution with power-law tails. Furthermore, the scale
is allowed to vary between the left and the right side of the peak. There is no
discontinuity at the maximum or elsewhere.

The generalized Crystal Ball distribution is often used to empirically model a
bell curve with heavier tails than a normal distribution. For a symmetric
distribution, the superior but less well-known alternative is the Student's
t-distribution. A superior asymmetric form also exists, called the non-central
Student's t-distribution - which is not implemented in numba-stats yet.

The Student's t distribution is superior, because it can be derived from an actual
statistical process, while the ad hoc Crystal Ball stitches unrelated distributions
together, a better name would be the 'Frankenstein distribution'. The construction
makes it numerically very difficult to fit, which has caused many a grievance among
practitioners, which could have been avoided by using the more stable Student's t.
"""

import numpy as np

from ._util import _generate_wrappers, _jit, _jit_pointwise, _prange
from .crystalball import (
    _log_density,
    _normal_integral,
    _normal_ppf,
    _powerlaw_integral,
    _powerlaw_ppf,
)

_doc_par = """
beta_left : float
    Distance from the mode in units of standard deviations where the Crystal Ball
    turns from a gaussian into a power law on the left side.
m_left : float
    Absolute value of the slope of the left powerlaw tail. Must be large than 1.
scale_left : float
    Standard deviation of the left side of the mode.
beta_right : float
    Distance from the mode in units of standard deviations where the Crystal Ball
    turns from a gaussian into a power law on the right side.
m_right : float
    Absolute value of the slope of the right powerlaw tail. Must be large than 1.
scale_right : float
    Standard deviation of the right side of the mode.
loc : float
    Location of the mode of the distribution.
"""


@_jit_pointwise(3)
def _norm_half(beta: float, m: float, scale: float) -> float:
    T = type(beta)
    return (_powerlaw_integral(-beta, beta, m) + _normal_integral(-beta, T(0))) * scale


@_jit(7)
def _logpdf(
    x: np.ndarray,
    beta_left: float,
    m_left: float,
    scale_left: float,
    beta_right: float,
    m_right: float,
    scale_right: float,
    loc: float,
) -> np.ndarray:
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    c = np.log(norm)
    r = np.empty_like(x)
    for i in _prange(len(r)):
        if x[i] < loc:
            beta = beta_left
            m = m_left
            z = (x[i] - loc) / scale_left
        else:
            beta = beta_right
            m = m_right
            z = (loc - x[i]) / scale_right
        r[i] = _log_density(z, beta, m) - c
    return r


@_jit(7)
def _pdf(
    x: np.ndarray,
    beta_left: float,
    m_left: float,
    scale_left: float,
    beta_right: float,
    m_right: float,
    scale_right: float,
    loc: float,
) -> np.ndarray:
    return np.exp(
        _logpdf(
            x,
            beta_left,
            m_left,
            scale_left,
            beta_right,
            m_right,
            scale_right,
            loc,
        )
    )


@_jit(7)
def _cdf(
    x: np.ndarray,
    beta_left: float,
    m_left: float,
    scale_left: float,
    beta_right: float,
    m_right: float,
    scale_right: float,
    loc: float,
) -> np.ndarray:
    T = type(beta_left)
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    r = np.empty_like(x)
    for i in _prange(len(x)):
        scale = T(1) / (scale_left if x[i] < loc else scale_right)
        z = (x[i] - loc) * scale
        if z < -beta_left:
            r[i] = _powerlaw_integral(z, beta_left, m_left) * scale_left / norm
        elif z < 0:
            r[i] = (
                (
                    _powerlaw_integral(-beta_left, beta_left, m_left)
                    + _normal_integral(-beta_left, z)
                )
                * scale_left
                / norm
            )
        elif z < beta_right:
            r[i] = (
                (
                    _powerlaw_integral(-beta_left, beta_left, m_left)
                    + _normal_integral(-beta_left, T(0))
                )
                * scale_left
                + _normal_integral(T(0), z) * scale_right
            ) / norm
        else:
            r[i] = (
                (
                    _powerlaw_integral(-beta_left, beta_left, m_left)
                    + _normal_integral(-beta_left, T(0))
                )
                * scale_left
                + (
                    _normal_integral(T(0), beta_right)
                    + _powerlaw_integral(-beta_right, beta_right, m_right)
                    - _powerlaw_integral(-z, beta_right, m_right)
                )
                * scale_right
            ) / norm
    return r


@_jit(7, cache=False)
def _ppf(
    p: np.ndarray,
    beta_left: float,
    m_left: float,
    scale_left: float,
    beta_right: float,
    m_right: float,
    scale_right: float,
    loc: float,
) -> np.ndarray:
    T = type(beta_left)
    norm = _norm_half(beta_left, m_left, scale_left) + _norm_half(
        beta_right, m_right, scale_right
    )
    pbeta_left = _powerlaw_integral(-beta_left, beta_left, m_left) * scale_left / norm
    pbeta_middle = (
        _powerlaw_integral(-beta_left, beta_left, m_left) * scale_left
        + _normal_integral(-beta_left, T(0)) * scale_left
    ) / norm
    pbeta_right = (
        _powerlaw_integral(-beta_left, beta_left, m_left) * scale_left
        + _normal_integral(-beta_left, T(0)) * scale_left
        + _normal_integral(T(0), beta_right) * scale_right
    ) / norm
    r = np.empty_like(p)
    for i in _prange(len(p)):
        if p[i] < pbeta_left:
            unnorm_p = p[i] * norm / scale_left
            z = _powerlaw_ppf(unnorm_p, beta_left, m_left)
            r[i] = loc + z * scale_left
        elif p[i] < pbeta_middle:
            left_powerlaw_contrib = _powerlaw_integral(-beta_left, beta_left, m_left)
            normal_integral_needed = (p[i] * norm / scale_left) - left_powerlaw_contrib
            z = _normal_ppf(normal_integral_needed, -beta_left)
            r[i] = loc + z * scale_left
        elif p[i] < pbeta_right:
            left_total = _norm_half(beta_left, m_left, scale_left)
            normal_integral_needed = (p[i] * norm - left_total) / scale_right
            z = _normal_ppf(normal_integral_needed, T(0))
            r[i] = loc + z * scale_right
        else:
            remaining_p = 1 - p[i]
            tail_contrib = remaining_p * norm / scale_right
            z = -_powerlaw_ppf(tail_contrib, beta_right, m_right)
            r[i] = loc + z * scale_right
    return r


_generate_wrappers(globals())
