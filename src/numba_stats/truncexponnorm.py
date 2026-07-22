"""
Truncated exponentially modified normal distribution.

The distribution is defined as the distribution of the sum of a normal and 
exponential random variate. This definition has a right tail, but a left
tailed version can be defined by taking the difference between a normal and
exponential random variate instead.

To convert between the left- and right-tailed versions, a coordinate
transformation x -> 2\mu - x can be used to reflect the x-axis.

https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
See Also
--------
scipy.stats.exponnorm: Untruncated scipy equivalent.
"""

from ._util import _jit, _generate_wrappers, _prange

import numpy as np

from . import exponnorm as _exponnorm

_doc_par = """
xmin : float
    Lower edge of the distribution.
xmax : float
    Upper edge of the distribution.
loc : float
    Location parameter.
scale : float
    Scale parameter.
tau : float
    Exponential decay parameter.
"""

@_jit(5)
def _pdf(x: np.ndarray, xmin: float, xmax: float, loc: float, scale: float, tau: float) -> np.ndarray:
    T = type(scale)
    pmin = _exponnorm._cdf1(xmin, loc, scale, tau)
    pmax = _exponnorm._cdf1(xmax, loc, scale, tau)
    r = np.zeros_like(x)
    for i in _prange(len(x)):
        if xmin <= x[i] < xmax:
            r[i] = _exponnorm._pdf1(x[i], loc, scale, tau) / (pmax - pmin)
        else:
            r[i] = T(0.0)
    return r

@_jit(5)
def _cdf(x: np.ndarray, xmin: float, xmax: float, loc: float, scale: float, tau: float) -> np.ndarray:
    T = type(scale)
    pmin = _exponnorm._cdf1(xmin, loc, scale, tau)
    pmax = _exponnorm._cdf1(xmax, loc, scale, tau)
    r = np.zeros_like(x)
    for i in _prange(len(x)):
            if xmin <= x[i]:
                if x[i] < xmax:
                    r[i] = (_exponnorm._cdf1(x[i], loc, scale, tau) - pmin) / (pmax - pmin)
                else:
                    r[i] = T(1.0)
            else:
                r[i] = T(0.0)
    return r

_generate_wrappers(globals())