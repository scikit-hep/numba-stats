"""
Noncentral chi squared distribution.

See Also
--------
scipy.stats.ncx2: Scipy equivalent.
"""

import numpy as np

from ._special import chndtr as _chndtr
from ._special import chndtrix as _chndtrix
from ._special import ive as _ive
from ._special import xlogy as _xlogy
from ._util import (
    _generate_wrappers,
    _jit,
    _jit_pointwise,
    _prange,
    _trans,
)

_doc_par = """
df: float
    Degrees of freedom.
nc: float
    Noncentrality parameter.
loc: float
    Location parameter.
scale: float
    Scale parameter.
"""


@_jit_pointwise(3, cache=False)
def _logpdf1(z: float, df: float, nc: float) -> float:
    # Implementation from scipy
    # https://github.com/scipy/scipy/blob/54ef5423f2e4376230ec3bfda6912a07a50958e3/scipy/stats/_continuous_distns.py#L7811
    T = type(z)
    two = T(2.0)
    df2 = df / two - T(1.0)
    zs = T(np.sqrt(z))
    ns = T(np.sqrt(nc))
    res = _xlogy(df2 / two, z / nc) - T(0.5) * (zs - ns) ** 2
    corr = _ive(df2, zs * ns) / two
    if corr > 0:
        return T(res + np.log(corr))
    else:
        return -T(np.inf)


@_jit_pointwise(3, cache=False)
def _cdf1(z: float, df: float, nc: float) -> float:
    T = type(z)
    return T(_chndtr(z, df, nc))


@_jit_pointwise(3, cache=False)
def _ppf1(p: float, df: float, nc: float) -> float:
    T = type(p)
    return T(_chndtrix(p, df, nc))


@_jit(4, cache=False)
def _logpdf(
    x: np.ndarray, df: float, nc: float, loc: float, scale: float
) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _logpdf1(r[i], df, nc) - np.log(scale)
    return r


@_jit(4, cache=False)
def _pdf(x: np.ndarray, df: float, nc: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, df, nc, loc, scale))


@_jit(4, cache=False)
def _cdf(x: np.ndarray, df: float, nc: float, loc: float, scale: float) -> np.ndarray:
    r = _trans(x, loc, scale)
    for i in _prange(len(r)):
        r[i] = _cdf1(r[i], df, nc)
    return r


@_jit(4, cache=False)
def _ppf(p: np.ndarray, df: float, nc: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(p)
    for i in _prange(len(r)):
        r[i] = scale * _ppf1(p[i], df, nc) + loc
    return r


_generate_wrappers(globals())
