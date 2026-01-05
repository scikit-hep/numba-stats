"""Utilities for code and docs generation to reduce boilerplate code."""

import math
import os
from typing import Any, Callable, Optional

import numba as nb
import numpy as np
from numba import prange as _prange  # noqa
from numba.core.errors import TypingError
from numba.extending import overload
from numba.types import Array

_Floats = (nb.float32, nb.float64)

__all__ = [
    "_prange",
    "_readonly_carray",
    "_jit_custom",
    "_jit_pointwise",
    "_jit",
    "_rvs_jit",
    "_seed",
    "_generate_wrappers",
    "_trans",
]

DistributionFunction = Callable[..., np.ndarray]


def _readonly_carray(T: type) -> Array:
    return Array(T, 1, "A", readonly=True)


def _jit_custom(
    signatures: Any, cache: bool = True
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Wrap numba.njit to reduce boilerplate code.

    We want to build jitted functions with explicit signatures to restrict the argument
    types which are used in the implemnetation to float32 or float64. We also want to
    pass specific options consistently: error_model='numpy' and inline='always'. The
    latter is important to profit from auto-parallelization of surrounding code.
    """
    return nb.njit(signatures, cache=cache, inline="always", error_model="numpy")  # type:ignore[no-any-return]


def _jit_pointwise(
    npar: int, *, cache: bool = True
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """
    Wrap numba.njit to reduce boilerplate code.

    We want to build jitted functions with explicit signatures to restrict the argument
    types which are used in the implemnetation to float32 or float64. We also want to
    pass specific options consistently: error_model='numpy' and inline='always'. The
    latter is important to profit from auto-parallelization of surrounding code.

    This decorator builds signatures with "narg" array arguments followed by "npar"
    scalar arguments, and it does that for the types float32 or float64.

    Parameters
    ----------
    npar : int
        Number of scalar arguments.
    cache : bool, optional (default: True)
        Whether to cache the compilation. We must turn this off if the function uses a
        function pointer from Scipy.
    """
    assert npar >= 0
    signatures = []
    for T in (nb.float32, nb.float64):
        sig = T(*([T] * npar))
        signatures.append(sig)
    return _jit_custom(signatures, cache=cache)


def _jit(
    npar: int, *, narg: int = 1, cache: bool = True
) -> Callable[[DistributionFunction], DistributionFunction]:
    """
    Wrap numba.njit to reduce boilerplate code.

    We want to build jitted functions with explicit signatures to restrict the argument
    types which are used in the implemnetation to float32 or float64. We also want to
    pass specific options consistently: error_model='numpy' and inline='always'. The
    latter is important to profit from auto-parallelization of surrounding code.

    This decorator builds signatures with "narg" array arguments followed by "npar"
    scalar arguments, and it does that for the types float32 or float64.

    Parameters
    ----------
    npar : int
        Number of scalar arguments.
    narg : int, optional (default: 1)
        Number of array arguments.
    cache : bool, optional (default: True)
        Whether to cache the compilation. We must turn this off if the function uses a
        function pointer from Scipy.
    """
    assert npar >= 0
    assert narg >= 1
    signatures = []
    for T in (nb.float32, nb.float64):
        sig = T[:](
            *[_readonly_carray(T) for _ in range(narg)], *[T for _ in range(npar)]
        )
        signatures.append(sig)
    return _jit_custom(signatures, cache=cache)


def _rvs_jit(
    arg: int, cache: bool = True
) -> Callable[[DistributionFunction], DistributionFunction]:
    signatures = []
    T = nb.float64  # nb.float32 cannot be supported
    # extra args at the end are for size and random_state
    sig = T[:](*[T for _ in range(arg)], nb.uint64, nb.optional(nb.uint64))
    signatures.append(sig)
    return _jit_custom(signatures, cache=cache)


@nb.njit(cache=True)  # type:ignore[misc]
def _seed(seed: Optional[int]) -> None:
    if seed is None:
        with nb.objmode(seed="optional(uint8)"):
            seed = np.frombuffer(os.urandom(8), dtype=np.uint64)[0]
    np.random.seed(seed)


def _wrap(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    def outer(first: np.ndarray, *rest: Any) -> np.ndarray:
        shape = np.shape(first)
        first = np.array(first).flatten()
        if first.dtype.kind != "f":
            first = first.astype(float)
        return fn(first, *rest).reshape(shape)

    return outer


@_jit(2)
def _trans(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    inv_scale = type(scale)(1) / scale
    return (x - loc) * inv_scale


@nb.njit(cache=True, inline="always", error_model="numpy")  # type:ignore[misc]
def _erf_inplace(x: np.ndarray) -> None:
    for i in _prange(len(x)):
        x[i] = math.erf(x[i])


@nb.njit(cache=True, inline="always", error_model="numpy")  # type:ignore[misc]
def _erfc_inplace(x: np.ndarray) -> None:
    for i in _prange(len(x)):
        x[i] = math.erfc(x[i])


def _type_check(first: Array, *rest: Any) -> None:
    if not (isinstance(first, Array) and first.dtype in _Floats):
        raise TypingError("first argument must be an array of floating point type")

    T = type(first.dtype)
    for i, tp in enumerate(rest):
        if not isinstance(tp, T):
            raise TypingError(f"argument {i + 1} must be of type {tp}")


def _generate_wrappers(d: dict[str, Any]) -> None:
    import inspect

    if "_wrap" not in d:
        d["_wrap"] = _wrap
    if "_type_check" not in d:
        d["_type_check"] = _type_check
    d["_overload"] = overload

    doc_par = d["_doc_par"].strip() if "_doc_par" in d else None

    for fname in (
        "pdf",
        "pmf",
        "logpdf",
        "logpmf",
        "cdf",
        "ppf",
        "density",
        "integral",
        "rvs",
    ):
        impl = f"_{fname}"
        if impl not in d:
            continue
        fn = d[impl]
        parameters = inspect.signature(fn).parameters
        args = ", ".join(parameters)
        args_with_types = ", ".join(
            str(x).replace("numpy", "np") for x in parameters.values()
        )
        doc_title = {
            "density": "Return density.",
            "integral": "Return integrated density.",
            "logpdf": "Return log of probability density.",
            "logpmf": "Return log of probability mass.",
            "pmf": "Return probability mass.",
            "pdf": "Return probability density.",
            "cdf": "Return cumulative probability.",
            "ppf": "Return quantile for given probability.",
            "rvs": "Return random samples from distribution.",
        }.get(fname, None)
        if fname == "ppf":
            before_par = """\
x: ArrayLike
    Probability. Must be between 0 and 1.
"""
        elif fname == "rvs":
            before_par = ""
        else:
            before_par = """\
x: ArrayLike
    Random variate.
"""
        if fname == "rvs":
            after_par = """
size : int
    Number of random variates.
random_state : int or None
    Seed of the random number generator. Default is None, which uses a random seed."""
        else:
            after_par = ""

        if fname == "rvs":
            code = f"""
def {fname}({args_with_types}):
    return {impl}({args})

@_overload({fname}, inline="always")
def _ol_{fname}({args_with_types}):
    return {impl}.__wrapped__
"""
        else:
            code = f"""
def {fname}({args_with_types}):
    return _wrap({impl})({args})

@_overload({fname}, inline="always")
def _ol_{fname}({args_with_types}):
    _type_check({args})
    return {impl}.__wrapped__
"""

        if doc_par is None:
            code += f"""
{fname}.__doc__ = {impl}.__doc__
"""
        else:
            assert doc_title is not None
            code += f"""
{fname}.__doc__ = \"\"\"
{doc_title}

Parameters
----------
{before_par}{doc_par}{after_par}

Returns
-------
Array-like
    Function evaluated at the x points.
\"\"\"
"""
        exec(code, d)
