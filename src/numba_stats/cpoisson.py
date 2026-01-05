"""
Continuous Poisson distribution.

The Poisson distribution is for discrete values, but the cdf can be generalised
to arbitrary real values. It was assumed that this distribution could be useful
to describe weighted data, but it is currently unclear how useful the
distribution is.

The pdf cannot be expressed in tabulated functions:

d G(x, mu)/d x = ln(mu) G(x, mu) + mu T(3, x, mu)

where G(x, mu) is the upper incomplete gamma function and T(m, s, x) is a
special case of the Meijer G-function, see
https://en.wikipedia.org/wiki/Incomplete_gamma_function#Derivatives

There is a Meijer G-function implemented in mpmath, but I don't know how to use it.
"""

import numpy as np

from ._special import gammaincc as _gammaincc
from ._util import _generate_wrappers, _jit, _prange

_doc_par = """
mu : float
    Expected value.
"""


@_jit(1, cache=False)
def _cdf(x: np.ndarray, mu: float) -> np.ndarray:
    r = np.empty_like(x)
    one = type(x[0])(1)
    for i in _prange(len(x)):
        r[i] = _gammaincc(x[i] + one, mu)
    return r


_generate_wrappers(globals())
