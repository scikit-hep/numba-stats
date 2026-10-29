r"""
Voigtian distribution.

This is the convolution of a Cauchy distribution with a normal distribution.

There is a closed form for the cdf in terms of a hypergeometric function, but it is
numerically unstable when the Cauchy width exceeds the normal width. Instead, with
z = (x - loc + i gamma) / (scale sqrt(2)), the cdf is computed for |z| < 6 by
Gauss-Legendre quadrature of the pdf from loc to x, and for |z| >= 6 from the
large-|z| expansion of the integrated Faddeeva function. The absolute error is below
1e-14 for gamma / scale from 1e-3 to 3e1.

https://en.wikipedia.org/wiki/Voigt_profile

See Also
--------
scipy.special.voigt_profile: Equvialent in Scipy.
"""

import cmath
import math

import numba as nb
import numpy as np

from ._special import voigt_profile as _voigt
from ._util import _generate_wrappers, _jit, _prange

_doc_par = """
gamma : float
    The half-width at half-maximum of the Cauchy distribution part.
loc : float
    Location of the mode.
scale : float
    Standard deviation of the normal distribution.
"""

# |z| at which the cdf switches from quadrature to the asymptotic expansion
_CDF_SWITCH = 6.0
# number of terms of the asymptotic expansion
_CDF_NTERMS = 24
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


@nb.njit(cache=True, inline="always", error_model="numpy")  # type:ignore[untyped-decorator]
def _cdf_expansion(z: complex) -> float:
    # Integrating the expansion w(t) ~ i / (sqrt(pi) t) sum_n (2n-1)!! / (2t^2)^n term
    # by term and fixing the constant by F(+inf) = 1 gives
    # F = 1 - arg(z) / pi + Im sum_{n>=1} (2n-1)!! / (2^(n+1) n z^(2n)) / pi.
    inv_z2 = 1.0 / (z * z)
    term = 0.25 * inv_z2  # n = 1
    total = 0j
    for n in range(1, _CDF_NTERMS + 1):
        total += term / n
        term *= (n + 0.5) * inv_z2
    if z.real < 0:
        # left of the mode, compute pi - arg(z) directly to keep relative precision
        leading = math.atan2(z.imag, -z.real) / math.pi
    else:
        leading = 1.0 - cmath.phase(z) / math.pi
    return leading + total.imag / math.pi


@_jit(3, cache=False)
def _pdf(x: np.ndarray, gamma: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(x)
    for i in _prange(len(x)):
        r[i] = _voigt(x[i] - loc, scale, gamma)
    return r


@_jit(3, cache=False)
def _cdf(x: np.ndarray, gamma: float, loc: float, scale: float) -> np.ndarray:
    # computed in double precision, since _voigt only supports double precision
    r = np.empty_like(x)
    inv = 1.0 / (np.float64(scale) * np.sqrt(2.0))
    for i in _prange(len(x)):
        d = np.float64(x[i]) - loc
        z = complex(d * inv, gamma * inv)
        if abs(z) >= _CDF_SWITCH:
            r[i] = _cdf_expansion(z)
        else:
            # one Gauss-Legendre panel on [loc, x], where the pdf is smooth
            half = 0.5 * d
            s = 0.0
            for k in range(len(_GL_NODES)):
                s += _GL_WEIGHTS[k] * _voigt(half * (_GL_NODES[k] + 1.0), scale, gamma)
            r[i] = 0.5 + half * s
    return r


_generate_wrappers(globals())
