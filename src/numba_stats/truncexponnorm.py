"""
Truncated exponentially modified normal distribution.

See the module exponnorm for a description of the untruncated distribution. There is
no scipy equivalent. The truncation parameters xmin and xmax follow the convention of
truncnorm and truncexpon.

See Also
--------
scipy.stats.exponnorm: Scipy equivalent of the untruncated distribution.
"""

import numpy as np

from . import exponnorm as _exponnorm
from ._util import _generate_wrappers, _jit, _prange

_doc_par = """
xmin : float
    Lower edge of the distribution.
xmax : float
    Upper edge of the distribution.
K : float
    Shape parameter. Mean of the exponential component divided by the standard
    deviation of the normal component. Must be positive.
loc : float
    Location of the mode of the normal component.
scale : float
    Standard deviation of the normal component.
"""


@_jit(5, cache=False)
def _logpdf(
    x: np.ndarray, xmin: float, xmax: float, K: float, loc: float, scale: float
) -> np.ndarray:
    T = type(scale)
    scale2 = T(1) / scale
    z = (x - loc) * scale2
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    c = np.log(scale * (_exponnorm._cdf1(zmax, K) - _exponnorm._cdf1(zmin, K)))
    for i in _prange(len(z)):
        if zmin <= z[i] < zmax:
            z[i] = _exponnorm._logpdf1(z[i], K) - c
        else:
            z[i] = -T(np.inf)
    return z


@_jit(5, cache=False)
def _pdf(
    x: np.ndarray, xmin: float, xmax: float, K: float, loc: float, scale: float
) -> np.ndarray:
    return np.exp(_logpdf(x, xmin, xmax, K, loc, scale))


@_jit(5, cache=False)
def _cdf(
    x: np.ndarray, xmin: float, xmax: float, K: float, loc: float, scale: float
) -> np.ndarray:
    T = type(scale)
    scale2 = T(1) / scale
    z = (x - loc) * scale2
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _exponnorm._cdf1(zmin, K)
    pmax = _exponnorm._cdf1(zmax, K)
    scale3 = T(1) / (pmax - pmin)
    for i in _prange(len(z)):
        if zmin <= z[i]:
            if z[i] < zmax:
                z[i] = (_exponnorm._cdf1(z[i], K) - pmin) * scale3
            else:
                z[i] = T(1)
        else:
            z[i] = T(0)
    return z


_generate_wrappers(globals())
