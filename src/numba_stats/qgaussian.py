"""
Q-Gaussian distribution.

A generalisation (q-analog) of the normal distribution based on Tsallis entropy. It
can be used an alternative model for the normal distribution to check for model bias.

It is equivalent to Student's t distribution and can be computed from the latter via
a change of variables, which is exploited in this implementation.

https://en.wikipedia.org/wiki/Q-Gaussian_distribution
"""

from typing import Optional

import numba as nb
import numpy as np

from . import norm as _norm
from . import t as _t
from ._util import _generate_wrappers, _jit, _rvs_jit

_doc_par = """
q : float
    Shape parameter between 1 and 3. For q = 1, the qgaussian is a normal distribution,
    for q == 3 it is a cauchy distribution.
loc : float
    Expectation value.
scale : float
    Width parameter.
"""


@nb.njit  # type:ignore[misc]
def _df_sigma(q: float, sigma: float) -> tuple[float, float]:
    # https://en.wikipedia.org/wiki/Q-Gaussian_distribution
    # relation to Student's t-distribution

    # 1/(2 sigma^2) = 1 / (3 - q)
    # 2 sigma^2 = 3 - q
    # sigma = sqrt((3 - q)/2)
    T = type(q)
    df = (T(3) - q) / (q - T(1))
    sigma /= np.sqrt(T(0.5) * (T(3) - q))

    return df, sigma


@_jit(3)
def _logpdf(x: np.ndarray, q: float, mu: float, sigma: float) -> np.ndarray:
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._logpdf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._logpdf(x, df, mu, sigma)


@_jit(3)
def _pdf(x: np.ndarray, q: float, mu: float, sigma: float) -> np.ndarray:
    return np.exp(_logpdf(x, q, mu, sigma))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, q: float, mu: float, sigma: float) -> np.ndarray:
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._cdf(x, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._cdf(x, df, mu, sigma)


@_jit(3, cache=False)
def _ppf(p: np.ndarray, q: float, mu: float, sigma: float) -> np.ndarray:
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._ppf(p, mu, sigma)

    df, sigma = _df_sigma(q, sigma)

    return _t._ppf(p, df, mu, sigma)


@_rvs_jit(3, cache=False)
def _rvs(
    q: float, mu: float, sigma: float, size: int, random_state: Optional[int]
) -> np.ndarray:
    if q < 1 or q > 3:
        raise ValueError("q < 1 or q > 3 are not supported")

    if q == 1:
        return _norm._rvs(mu, sigma, size, random_state)

    df, sigma = _df_sigma(q, sigma)

    return _t._rvs(df, mu, sigma, size, random_state)


_generate_wrappers(globals())
