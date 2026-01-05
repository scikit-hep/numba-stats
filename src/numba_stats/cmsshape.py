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

import numpy as np

from ._util import _erf_inplace, _erfc_inplace, _generate_wrappers, _jit

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


@_jit(3)
def _cdf(x: np.ndarray, beta: float, gamma: float, loc: float) -> np.ndarray:
    T = type(beta)
    y = x - loc
    two = T(2)
    half = T(0.5)
    t1 = gamma / (two * beta) + beta * y
    _erf_inplace(t1)
    t2 = np.exp(-((gamma / (two * beta)) ** two) - gamma * y)
    t3 = -beta * y
    _erfc_inplace(t3)
    return half * (t1 - t2 * t3) + half  # type:ignore[no-any-return]


_generate_wrappers(globals())
