"""
Student's t distribution.

See Also
--------
scipy.stats.t: Scipy equivalent.
"""

from math import lgamma as _lgamma

import numpy as np

from ._special import stdtr as _stdtr
from ._special import stdtrit as _stdtrit
from ._util import _generate_wrappers, _jit, _prange, _rvs_jit, _seed, _trans

_doc_par = """
df : float
    Degrees of freedom.
loc : float
    Location of the mode.
scale : float
    Width parameter.
"""


@_jit(3, cache=False)
def _logpdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    T = type(df)
    z = _trans(x, loc, scale)
    k = T(0.5) * (df + T(1))
    c = _lgamma(k) - _lgamma(T(0.5) * df)
    c -= T(0.5) * np.log(df * T(np.pi))
    c -= np.log(scale)
    for i in _prange(len(z)):
        z[i] = -k * np.log(T(1) + (z[i] * z[i]) / df) + c
    return z


@_jit(3, cache=False)
def _pdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, df, loc, scale))


@_jit(3, cache=False)
def _cdf(x: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    z = _trans(x, loc, scale)
    for i in _prange(len(z)):
        z[i] = _stdtr(df, z[i])
    return z


@_jit(3, cache=False)
def _ppf(p: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    T = type(df)
    r = np.empty_like(p)
    for i in _prange(len(p)):
        if p[i] == 0:
            r[i] = -T(np.inf)
        elif p[i] == 1:
            r[i] = T(np.inf)
        else:
            r[i] = _stdtrit(df, p[i])
    return scale * r + loc


@_rvs_jit(3)
def _rvs(
    df: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    return loc + scale * np.random.standard_t(df, size)


_generate_wrappers(globals())
