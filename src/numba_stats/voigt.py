"""
Voigtian distribution.

This is the convolution of a Cauchy distribution with a normal distribution.

There is a closed form for the cdf, but the required hypergeometric function is not
implemented anywhere.

https://en.wikipedia.org/wiki/Voigt_profile

See Also
--------
scipy.special.voigt_profile: Equvialent in Scipy.
"""

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


@_jit(3, cache=False)
def _pdf(x: np.ndarray, gamma: float, loc: float, scale: float) -> np.ndarray:
    r = np.empty_like(x)
    for i in _prange(len(x)):
        r[i] = _voigt(x[i] - loc, scale, gamma)
    return r


_generate_wrappers(globals())
