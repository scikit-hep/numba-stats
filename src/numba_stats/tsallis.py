"""
Tsallis-Hagedorn distribution.

A generalisation (q-analog) of the exponential distribution based on Tsallis entropy. It
approximately describes the pT distribution charged particles produced in high-energy
minimum bias particle collisions.
"""

import numpy as np

from ._util import _generate_wrappers, _jit, _jit_pointwise, _prange

_doc_par = """
m : float
    Mass of the particle.
t :  float
    Width parameter.
n : float
    Absolute value of the exponent of the power law.
"""


@_jit(3)
def _pdf(x: np.ndarray, m: float, t: float, n: float) -> np.ndarray:
    # Formula from CMS, Eur. Phys. J. C (2012) 72:2164
    if n <= 2:
        raise ValueError("n > 2 is required")

    T = type(m)
    mt = np.sqrt(m * m + x * x)
    nt = n * t
    c = (n - T(1)) * (n - T(2)) / (nt * (nt + (n - T(2)) * m))
    return c * x * (T(1) + (mt - m) / nt) ** -n  # type:ignore[no-any-return]


@_jit_pointwise(4)
def _cdf1(x: float, m: float, t: float, n: float) -> float:
    T = type(m)
    mt = np.sqrt(m * m + x * x)
    nt = n * t
    # Formula computed from tsallis_pdf with Sympy, then simplified by hand
    return (  # type:ignore[no-any-return]
        ((mt - m) / nt + T(1)) ** (T(1) - n)
        * (m + mt - n * (mt + t))
        / (m * (n - T(2)) + nt)
    )


@_jit(3)
def _cdf(x: np.ndarray, m: float, t: float, n: float) -> np.ndarray:
    if n <= 2:
        raise ValueError("n > 2 is required")
    r = np.empty_like(x)
    for i in _prange(len(r)):
        r[i] = _cdf1(x[i], m, t, n)
    return r


@_jit_pointwise(5)
def _integrate(lo: float, hi: float, m: float, t: float, n: float) -> float:
    if n <= 2:
        raise ValueError("n > 2 is required")
    return _cdf1(hi, m, t, n) - _cdf1(lo, m, t, n)


_generate_wrappers(globals())
