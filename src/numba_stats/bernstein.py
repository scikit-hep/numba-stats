"""
Empirical density distribution formed by a Bernstein polynomial.

The Bernstein polynomial basis is better suited to model a probability distribution
than the Chebychev basis, since it is possible to implement the constraint
f(x; p) >= 0 with simple parameter limits p >= 0 (where p is a vector).

The density function and its integral are not normalised. Normalising would create
several issues while providing no practical benefit. Use this function in an extended
maximum-likelihood fit and do not multiply it with a scale. The scale is implicit in the
parameter vector beta. If you really need to know the scale after the fit (usually you
do not), use :func:`integral` to compute it.

See Also
--------
scipy.interpolate.BPoly: Bernstein polynomials in Scipy.
"""

from collections.abc import Iterable
from typing import Any, Callable

import numpy as np

from ._util import (
    _Floats,
    _generate_wrappers,
    _jit,
    _trans,
)


@_jit(0, narg=2)
def _de_castlejau(z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # De Casteljau algorithm, numerically stable
    n = len(beta)
    res = np.full_like(z, np.nan)
    betai = np.empty_like(beta)
    # not sure how to parallelize this, each worker thread needs its own betai
    for i in range(len(z)):
        betai[:] = beta
        azi = 1.0 - z[i]
        for j in range(1, n):
            for k in range(n - j):
                betai[k] = betai[k] * azi + betai[k + 1] * z[i]
        res[i] = betai[0]
    return res


@_jit(0)
def _beta_int(beta: np.ndarray) -> np.ndarray:
    n = len(beta)
    r = np.zeros(n + 1, dtype=beta.dtype)
    for j in range(1, n + 1):
        for k in range(j):
            r[j] += beta[k]
    r *= 1.0 / n
    return r


@_jit(2, narg=2)
def _density(x: np.ndarray, beta: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """
    Return density described by a Bernstein polynomial.

    The function is non-negative, if each element of the parameter vector beta is non-
    negative, see module description.

    This function is by design equivalent to
    ``scipy.interpolate.BPoly(np.array(beta)[:, np.newaxis], [x[0], x[-1]])(x)``.

    Parameters
    ----------
    x : ArrayLike
        Values where the density is evaluated.
    beta : ArrayLike
        Vector of parameters (1D).
    xmin : float
        Lower edge of the domain of x.
    xmax : float
        Upper edge of the domain of x.

    Returns
    -------
    ndarray
        Function values.

    Examples
    --------
    >>> import numpy as np
    >>> from numba_stats import bernstein
    >>> x = np.linspace(-1, 1)
    >>> y = bernstein.density(x, [1, 2], -1, 1)

    See Also
    --------
    scipy.interpolate.BPoly
    """
    z = _trans(x, xmin, xmax - xmin)
    return _de_castlejau(z, beta)


@_jit(2, narg=2)
def _integral(x: np.ndarray, beta: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """
    Return integral of a Bernstein polynomial from xmin to x.

    Parameters
    ----------
    x : ArrayLike
        Values up to which the integral is computed, starting from xmin.
    beta : ArrayLike
        Vector of parameters (1D).
    xmin : float
        Lower edge of the domain of x.
    xmax : float
        Upper edge of the domain of x.

    Returns
    -------
    ndarray
        Integral values.

    Examples
    --------
    >>> import numpy as np
    >>> from numba_stats import bernstein
    >>> x = np.linspace(-1, 1)
    >>> y = bernstein.integral(x, [1, 2], -1, 1)

    See Also
    --------
    scipy.interpolate.BPoly
    """
    scale = xmax - xmin
    z = _trans(x, xmin, scale)
    beta = _beta_int(beta) * scale
    return _de_castlejau(z, beta)


def _wrap(
    fn: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
) -> Callable[[np.ndarray, np.ndarray, float, float], np.ndarray]:
    def outer(x: np.ndarray, beta: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
        shape = np.shape(x)

        def process(arg: Iterable[float]) -> np.ndarray:
            arg = np.array(arg).flatten()
            if arg.dtype.kind != "f":
                arg = arg.astype(float)
            return arg

        x = process(x)
        beta = process(beta)
        return fn(x, beta, xmin, xmax).reshape(shape)

    return outer


def _type_check(x: Any, beta: Any, xmin: Any, xmax: Any) -> None:
    from numba.core.errors import TypingError
    from numba.types import Array

    for arg in (x, beta):
        if not (isinstance(arg, Array) and arg.dtype in _Floats):
            raise TypingError(
                "first two arguments must be arrays of floating point type"
            )
    T = type(arg.dtype)
    for i, tp in enumerate((xmin, xmax)):
        if not isinstance(tp, T):
            raise TypingError(f"argument {i + 1} must be of type {tp}")


_generate_wrappers(globals())


def __getattr__(key: str) -> Any:
    # Temporary hack to maintain backward compatibility
    import warnings

    if key in ("scaled_pdf", "scaled_cdf"):
        r = {"scaled_pdf": "density", "scaled_cdf": "integral"}
        warnings.warn(
            f"bernstein.{key} is deprecated and will be removed in a future release, "
            f"use bernstein.{r[key]} instead",
            FutureWarning,
            1,
        )
        return globals()[r[key]]
    raise AttributeError
