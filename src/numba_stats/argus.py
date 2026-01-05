"""
Generalised ARGUS distribution.

The ARGUS distribution is named after the particle physics experiment ARGUS and it
describes the reconstructed invariant mass of a decayed particle candidate
in continuum background.
It is motivated from experimental observation. Here we have the generalised version
of the ARGUS distribution that can be used to describe a more peaking like distribution.
p = 0.5 gives the normal ARGUS distribution.

https://en.wikipedia.org/wiki/ARGUS_distribution

See Also
--------
scipy.stats.argus: Scipy equivalent.
"""

from math import lgamma as _lg

import numpy as np

from ._special import gammainc as _ginc
from ._util import _generate_wrappers, _jit, _prange

_doc_par = """
x : Array-like
    Random variate, between 0 and c.
chi : float
    Must be larger than 0 and represents the cutoff.
c : float
    Must be larger than 0 and represents the curvature.
p : float
    Must be larger than -1 and represents the power.
"""


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, chi: float, c: float, p: float) -> np.ndarray:
    T = type(p)
    one = T(1)
    two = T(2)
    half = T(0.5)
    half_chi2 = half * chi * chi
    inv_c2 = one / (c * c)
    p1 = p + one
    r = np.empty_like(x)
    for i in _prange(len(x)):
        xi = x[i]
        if 0 <= xi and xi <= c:
            y = one - xi * xi * inv_c2
            r[i] = (
                -half_chi2 * y
                + p * (np.log(y) - np.log(two))
                + two * (p1 * np.log(chi) - np.log(c))
                + np.log(xi)
                - T(_lg(p1))
                - np.log(T(_ginc(p1, half_chi2)))
            )
        else:
            r[i] = -np.inf
    return r


@_jit(3, cache=False)
def _pdf(x: np.ndarray, chi: float, c: float, p: float) -> np.ndarray:
    return np.exp(_logpdf(x, chi, c, p))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, chi: float, c: float, p: float) -> np.ndarray:
    T = type(p)
    zero = T(0)
    one = T(1)
    half = T(0.5)
    p1 = p + one
    half_chi2 = half * chi * chi
    inv_c2 = one / (c * c)
    inv_ginc = one / T(_ginc(p1, half_chi2))
    r = np.empty_like(x)
    for i in _prange(len(x)):
        xi = x[i]
        if 0 <= xi:
            if xi <= c:
                y = one - xi * xi * inv_c2
                r[i] = one - T(_ginc(p1, half_chi2 * y)) * inv_ginc
            else:
                r[i] = one
        else:
            r[i] = zero
    return r


_generate_wrappers(globals())
