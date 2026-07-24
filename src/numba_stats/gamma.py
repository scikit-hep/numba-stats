"""
Gamma distribution.

See Also
--------
scipy.stats.gamma: Scipy equivalent.
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
    _trans,
)

_doc_par = """
alpha: float
    Shape parameter.
loc: float
    Location parameter.
scale: float
    Scale parameter.
"""


@_jit_pointwise(2, cache=False)
def _logpdf1(z: float, alpha: float) -> float:
    T = type(z)
    return (alpha - T(1)) * T(np.log(z)) - z - T(_gammaln(alpha))


@_jit_pointwise(2, cache=False)
def _cdf1(z: float, alpha: float) -> float:
    T = type(z)
    return T(1) - T(_gammaincc(alpha, z))


@_jit_pointwise(2, cache=False)
def _ppf1(p: float, alpha: float) -> float:
    T = type(p)
    return T(_gammaincinv(alpha, p))


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, alpha: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _logpdf1(r[i], alpha) - np.log(scale)
    return r


@_jit(3, cache=False)
def _pdf(x: np.ndarray, alpha: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, alpha, loc, scale))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, alpha: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _cdf1(r[i], alpha)
    return r


@_jit(3, cache=False)
def _ppf(p: np.ndarray, alpha: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i], alpha) + loc
    return r


_generate_wrappers(globals())
