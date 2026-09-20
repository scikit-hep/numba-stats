"""
Novosibirsk distribution.

An asymmetric peak which is obtained from a normal distribution by replacing the variate
with the logarithm of a linear function of the variate. It is used in particle physics
to model radiative tails, for example, in the invariant mass distribution of a
resonance.

The support is bounded on one side, where the argument of the logarithm vanishes. The
location parameter is the mode of the distribution and the scale parameter is the
standard deviation of the normal distribution with the same full width at half maximum.

Formula taken from H. Ikeda et al. NIM A441 (2000), p. 401 (Belle Collaboration).

Notes
-----
This implementation was modeled after
https://root.cern.ch/doc/master/RooNovosibirsk_8cxx_source.html, but the density is
normalized to unity and the cdf and the ppf are computed analytically. The parameters
"peak", "width", and "tail" of the original implementation are called "loc", "scale",
and "lambd" here to follow the conventions of this library.
"""

from math import asinh as _asinh
from math import erfc as _erfc
from math import expm1 as _expm1
from math import log1p as _log1p

import numpy as np

from . import norm as _norm
from ._special import ndtri as _ndtri
from ._util import _generate_wrappers, _jit, _jit_pointwise, _prange, _rvs_jit, _seed

_doc_par = """
lambd : float
    Tail parameter. The distribution is skewed to the left for positive values and to
    the right for negative values. For lambd = 0, it is a normal distribution.
loc : float
    Location of the mode of the distribution.
scale : float
    Width parameter. It is the standard deviation of the normal distribution with the
    same full width at half maximum.
"""


@_jit_pointwise(1)
def _width_zero(lambd: float) -> float:
    # width of the Gaussian in log-space, approaches lambd for lambd -> 0
    T = type(lambd)
    c = T(np.sqrt(np.log(4)))
    return T(_asinh(lambd * c)) / c


@_jit(3)
def _logpdf(x: np.ndarray, lambd: float, loc: float, scale: float) -> np.ndarray:
    if lambd == 0:
        return _norm._logpdf(x, loc, scale)
    T = type(lambd)
    half = T(0.5)
    s = _width_zero(lambd)
    c = np.log(abs(lambd) / (abs(s) * scale)) - half * (s * s + T(np.log(2 * np.pi)))
    r = np.empty_like(x)
    for i in _prange(len(r)):
        u = -lambd * (x[i] - loc) / scale
        if u <= -1:
            # argument of the logarithm is negative, density is zero
            r[i] = -np.inf
        else:
            v = T(_log1p(u)) / s
            r[i] = c - half * v * v
    return r


@_jit(3)
def _pdf(x: np.ndarray, lambd: float, loc: float, scale: float) -> np.ndarray:
    return np.exp(_logpdf(x, lambd, loc, scale))


@_jit(3)
def _cdf(x: np.ndarray, lambd: float, loc: float, scale: float) -> np.ndarray:
    if lambd == 0:
        return _norm._cdf(x, loc, scale)
    T = type(lambd)
    zero = T(0)
    one = T(1)
    half = T(0.5)
    c = T(np.sqrt(0.5))
    s = _width_zero(lambd)
    r = np.empty_like(x)
    for i in _prange(len(r)):
        u = -lambd * (x[i] - loc) / scale
        if u <= -1:
            # support is bounded from above for lambd > 0 and from below otherwise
            r[i] = one if lambd > 0 else zero
        else:
            r[i] = half * T(_erfc((T(_log1p(u)) / s - s) * c))
    return r


@_jit(3, cache=False)  # cannot cache because of _ndtri
def _ppf(p: np.ndarray, lambd: float, loc: float, scale: float) -> np.ndarray:
    if lambd == 0:
        return _norm._ppf(p, loc, scale)
    T = type(lambd)
    s = _width_zero(lambd)
    r = np.empty_like(p)
    for i in _prange(len(r)):
        z = -T(_expm1(s * (s - T(_ndtri(p[i]))))) / lambd
        r[i] = z * scale + loc
    return r


@_rvs_jit(3, cache=False)
def _rvs(
    lambd: float, loc: float, scale: float, size: int, random_state: int | None
) -> np.ndarray:
    _seed(random_state)
    p = np.random.uniform(0, 1, size)
    return _ppf(p, lambd, loc, scale)


_generate_wrappers(globals())
