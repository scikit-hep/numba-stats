"""
Chi squared distribution.

See Also
--------
scipy.stats.chi2: Scipy equivalent.
"""

import numpy as np

from ._special import gammaincc as _gammaincc
from ._special import gammaincinv as _gammaincinv
from ._special import gammaln as _gammaln
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
df: float
    Degrees of freedom.
loc: float
    Location parameter.
scale: float
    Scale parameter.
"""


@_jit_pointwise(2, cache=False)
def _logpdf1(z: float, df: float) -> float:
    T = type(z)
    half = T(0.5)
    log2 = T(np.log(2))
    return (
        (half * df - T(1)) * T(np.log(z))
        - half * z
        - T(_gammaln(half * df))
        - half * df * log2
    )


@_jit_pointwise(2, cache=False)
def _cdf1(z: float, df: float) -> float:
    T = type(z)
    half = T(0.5)
    return T(1) - T(_gammaincc(half * df, half * z))


@_jit_pointwise(2, cache=False)
def _ppf1(p: float, df: float) -> float:
    T = type(p)
    return T(2.0 * _gammaincinv(0.5 * df, p))


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _logpdf1(r[i], df) - np.log(scale)
    return r


@_jit(3, cache=False)
def _pdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, df, loc, scale))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _cdf1(r[i], df)
    return r


@_jit(3, cache=False)
def _ppf(p: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i], df) + loc
    return r


@_rvs_jit(3, cache=False)
def _rvs(
    df: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    # Inverse transform sampling
    u = np.random.uniform(0, 1, size)
    return _ppf(u, df, loc, scale)


_generate_wrappers(globals())
