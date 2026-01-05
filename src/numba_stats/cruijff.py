"""
Cruijff distribution.

See For example: https://arxiv.org/abs/1005.4087
"""

import numpy as np

from ._util import _generate_wrappers, _jit, _prange

_doc_par = """
beta_left: float
    Left tail acceleration parameter.
beta_right: float
    Right tail acceleration parameter.
loc : float
    Location of the maximum of the distribution.
scale_left : float
    Left width parameter.
scale_right: float
    Right width parameter.
"""


@_jit(5)
def _density(
    x: np.ndarray,
    beta_left: float,
    beta_right: float,
    loc: float,
    scale_left: float,
    scale_right: float,
) -> np.ndarray:
    r = np.empty_like(x)
    for i in _prange(len(x)):
        if x[i] < loc:
            scale = scale_left
            beta = beta_left
        else:
            scale = scale_right
            beta = beta_right
        z = (x[i] - loc) / scale
        r[i] = -0.5 * z**2 / (1 + beta * z**2)
    return np.exp(r)  # type:ignore[no-any-return]


_generate_wrappers(globals())
