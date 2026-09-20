"""
Truncated normal distribution.

See Also
--------
scipy.stats.truncnorm: Scipy equivalent.
"""

import numpy as np

from . import norm as _norm
from ._util import _generate_wrappers, _jit, _prange, _rvs_jit, _seed

_doc_par = """
xmin : float
    Lower edge of the distribution.
xmax : float
    Upper edge of the distribution.
loc : float
    Location of the mode.
scale : float
    Width parameter.
"""


@_jit(4)
def _logpdf(
    x: np.ndarray, xmin: float, xmax: float, loc: float, scale: float
) -> np.ndarray:
    T = type(scale)
    scale2 = T(1) / scale
    z = (x - loc) * scale2
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    scale *= _norm._cdf1(zmax) - _norm._cdf1(zmin)
    for i in _prange(len(z)):
        if zmin <= z[i] < zmax:
            z[i] = _norm._logpdf1(z[i]) - np.log(scale)
        else:
            z[i] = -T(np.inf)
    return z


@_jit(4)
def _pdf(
    x: np.ndarray, xmin: float, xmax: float, loc: float, scale: float
) -> np.ndarray:
    return np.exp(_logpdf(x, xmin, xmax, loc, scale))


@_jit(4)
def _cdf(
    x: np.ndarray, xmin: float, xmax: float, loc: float, scale: float
) -> np.ndarray:
    scale = type(scale)(1) / scale
    r = (x - loc) * scale
    zmin = (xmin - loc) * scale
    zmax = (xmax - loc) * scale
    pmin = _norm._cdf1(zmin)
    pmax = _norm._cdf1(zmax)
    for i in _prange(len(r)):
        if zmin <= r[i]:
            if r[i] < zmax:
                r[i] = (_norm._cdf1(r[i]) - pmin) / (pmax - pmin)
            else:
                r[i] = 1.0
        else:
            r[i] = 0.0
    return r


@_jit(4, cache=False)
def _ppf(
    p: np.ndarray, xmin: float, xmax: float, loc: float, scale: float
) -> np.ndarray:
    scale2 = type(scale)(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _norm._cdf1(zmin)
    pmax = _norm._cdf1(zmax)
    r = p * (pmax - pmin) + pmin
    for i in _prange(len(r)):
        r[i] = _norm._ppf1(r[i])
    return scale * r + loc


@_rvs_jit(4, cache=False)
def _rvs(
    xmin: float,
    xmax: float,
    loc: float,
    scale: float,
    size: int,
    random_state: int | None,
) -> np.ndarray:
    _seed(random_state)
    p = np.random.uniform(0, 1, size)
    return _ppf(p, xmin, xmax, loc, scale)


_generate_wrappers(globals())
