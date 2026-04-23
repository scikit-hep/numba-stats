"""
Truncated exponential distribution.

See Also
--------
scipy.stats.truncexpon: Scipy equivalent.
"""

import numpy as np

from . import expon as _expon
from ._util import _generate_wrappers, _jit, _prange, _rvs_jit, _seed, _trans

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
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    c = np.log(scale * (_expon._cdf1(zmax) - _expon._cdf1(zmin)))
    zmin = max(zmin, T(0))
    for i in _prange(len(z)):
        if zmin <= z[i] < zmax:
            z[i] = -z[i] - c
        else:
            z[i] = -np.inf
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
    T = type(xmin)
    z = _trans(x, loc, scale)
    scale2 = T(1) / scale
    zmin = (xmin - loc) * scale2
    zmax = (xmax - loc) * scale2
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    scale3 = T(1) / (pmax - pmin)
    zmin = max(zmin, T(0))
    for i in _prange(len(z)):
        if zmin <= z[i]:
            if z[i] < zmax:
                z[i] = (_expon._cdf1(z[i]) - pmin) * scale3
            else:
                z[i] = 1
        else:
            z[i] = 0
    return z


@_jit(4)
def _ppf(
    p: np.ndarray, xmin: float, xmax: float, loc: float, scale: float
) -> np.ndarray:
    zmin = (xmin - loc) / scale
    zmax = (xmax - loc) / scale
    pmin = _expon._cdf1(zmin)
    pmax = _expon._cdf1(zmax)
    z = p * (pmax - pmin) + pmin
    for i in _prange(len(z)):
        z[i] = _expon._ppf1(z[i])
    return z * scale + loc


@_rvs_jit(4)
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
