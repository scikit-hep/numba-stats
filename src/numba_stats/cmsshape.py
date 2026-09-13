"""
CMS-Shape distribution (for lack of a better name).

The distribution consists of an exponential decay suppressed at small values by the
complementary error function. The product is an asymmetric peak with a bell shape on the
left-hand side and an exponential tail on the right-hand side. This shape is used by the
CMS experiment to model the background in the invariant mass distribution of Z to ll
decay candidates.

Notes
-----
This implementation was modeled after
https://gitlab.cern.ch/cms-muonPOG/spark_tnp/-/blob/Spark3/RooCMSShape.cc, but heavily
modified. An analytical normalization and an analytical cdf were added. The parameters
"alpha" and "peak" in the original implementation turned out to be redundant and have
been replaced with a single parameter "loc", which is the approximate center of the
distribution.
"""

from math import erf as _erf
from math import erfc as _erfc

import numpy as np

from ._util import _erfc_inplace, _generate_wrappers, _jit, _jit_pointwise, _prange

_doc_par = """
beta : float
    Steepness of the error function. Must be positive.
gamma : float
    Steepness of the exponential distribution. Must be positive.
loc: float
    Approximate center of the distribution.
"""


@_jit(3)
def _logpdf(x: np.ndarray, beta: float, gamma: float, loc: float) -> np.ndarray:
    T = type(beta)
    two = T(2)
    half = T(0.5)
    v = -(x - loc) * beta
    _erfc_inplace(v)
    u = (x - loc) * gamma
    T = type(beta)
    log_t = (half * gamma / beta) ** two
    return np.log(v) - u + np.log(half * gamma) - log_t  # type:ignore[no-any-return]


@_jit(3)
def _pdf(x: np.ndarray, beta: float, gamma: float, loc: float) -> np.ndarray:
    return np.exp(_logpdf(x, beta, gamma, loc))


@_jit_pointwise(3)
def _cdf1(y: float, beta: float, gamma: float) -> float:
    T = type(y)
    two = T(2)
    half = T(0.5)
    g2b = gamma / (two * beta)
    return (  # type:ignore[no-any-return]
        half
        * (
            T(_erf(g2b + beta * y))
            - np.exp(-g2b * g2b - gamma * y) * T(_erfc(-beta * y))
        )
        + half
    )


@_jit(3)
def _cdf(x: np.ndarray, beta: float, gamma: float, loc: float) -> np.ndarray:
    r = np.empty_like(x)
    for i in _prange(len(r)):
        r[i] = _cdf1(x[i] - loc, beta, gamma)
    return r


@_jit_pointwise(5)
def _integrate(lo: float, hi: float, beta: float, gamma: float, loc: float) -> float:
    return _cdf1(hi - loc, beta, gamma) - _cdf1(lo - loc, beta, gamma)


_generate_wrappers(globals())
