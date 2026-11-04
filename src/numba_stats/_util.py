"""Utilities for code and docs generation to reduce boilerplate code."""

import math
import os
from collections.abc import Callable
from typing import Any

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


@nb.njit(cache=True)  # type: ignore[untyped-decorator]
def _seed(seed: int | None) -> None:
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


@nb.njit(cache=True, inline="always", error_model="numpy")  # type:ignore[untyped-decorator]
def _erf_inplace(x: np.ndarray) -> None:
    for i in _prange(len(x)):
        x[i] = math.erf(x[i])


@nb.njit(cache=True, inline="always", error_model="numpy")  # type:ignore[untyped-decorator]
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


def _type_check_scalar(lo: Any, hi: Any, *rest: Any) -> Any:
    # parameters must be floats of the same type, lo and hi may also be integers, in
    # which case the numpy type to cast them to is returned
    T = rest[0]
    for tp in rest:
        if tp != T or not isinstance(T, nb.types.Float):
            raise TypingError("parameters must be floats of the same type")
    cast = False
    for tp in (lo, hi):
        if isinstance(tp, nb.types.Integer):
            cast = True
        elif tp != T:
            raise TypingError("lo and hi must be integers or floats of parameter type")
    if cast:
        return np.float32 if T == nb.types.float32 else np.float64
    return None


def _asarray(x: Any) -> np.ndarray:
    x = np.asarray(x)
    return x if x.dtype.kind == "f" else x.astype(float)


def _generate_wrappers(d: dict[str, Any]) -> None:
    import inspect

    if "_wrap" not in d:
        d["_wrap"] = _wrap
    if "_type_check" not in d:
        d["_type_check"] = _type_check
    if "_type_check_scalar" not in d:
        d["_type_check_scalar"] = _type_check_scalar
    d["_asarray"] = _asarray
    d["_overload"] = overload
    d["_np"] = np
    d["_nb_types"] = nb.types

    doc_par = d["_doc_par"].strip() if "_doc_par" in d else None

    # integrate is the difference of the cumulative function at the two limits,
    # computed in compiled code to avoid the overhead of the array interface
    src = "_cdf" if "_cdf" in d else "_integral"
    if src in d and "_integrate" not in d:
        _, *rest = inspect.signature(d[src]).parameters.values()
        # distributions with array parameters have to implement _integrate
        if all(p.annotation is float for p in rest):
            d["_np"] = np
            d["_jit_pointwise"] = _jit_pointwise
            names = ", ".join(p.name for p in rest)
            names_with_types = ", ".join(f"{p.name}: float" for p in rest)
            exec(
                f"""
@_jit_pointwise({len(rest) + 2}, cache=False)
def _integrate(lo: float, hi: float, {names_with_types}) -> float:
    _x = _np.empty(2, type(lo))
    _x[0] = lo
    _x[1] = hi
    _r = {src}(_x, {names})
    return _r[1] - _r[0]
""",
                d,
            )

    for fname in (
        "pdf",
        "pmf",
        "logpdf",
        "logpmf",
        "cdf",
        "ppf",
        "density",
        "integral",
        "integrate",
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
            "integrate": (
                "Return probability mass summed over an interval."
                if "_pmf" in d
                else "Return integral of density over an interval."
                if "_density" in d
                else "Return integral of probability density over an interval."
            ),
        }.get(fname, None)
        if fname == "ppf":
            before_par = """\
x: ArrayLike
    Probability. Must be between 0 and 1.
"""
        elif fname == "rvs":
            before_par = ""
        elif fname == "integrate" and "_pmf" in d:
            before_par = """\
lo : float
    Lower limit, excluded.
hi : float
    Upper limit, included.
"""
        elif fname == "integrate":
            before_par = """\
lo : float
    Lower limit of the integral.
hi : float
    Upper limit of the integral.
"""
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
        elif fname == "integrate":
            lo, hi, *rest = parameters
            # array parameters are converted like the first argument of _wrap
            conv = "".join(
                f"    {k} = _asarray({k})\n"
                for k, v in parameters.items()
                if v.annotation is np.ndarray
            )
            code = f"""
def {fname}({args_with_types}):
{conv}    return {impl}({args})

@_overload({fname}, inline="always")
def _ol_{fname}({args_with_types}):
    dt = _type_check_scalar({args})
    if dt is None:
        return {impl}.__wrapped__

    def impl({args_with_types}):
        return {impl}(dt({lo}), dt({hi}), {", ".join(rest)})

    return impl
"""
        else:
            first, *rest = parameters
            rest_args = ", ".join(rest)
            code = f"""
def {fname}({args_with_types}):
    return _wrap({impl})({args})

@_overload({fname}, inline="always")
def _ol_{fname}({args_with_types}):
    if isinstance({first}, (_nb_types.Float, _nb_types.Integer)):
        # scalar variate: evaluate on an array of length 1 and return the element
        T = {first} if isinstance({first}, _nb_types.Float) else _nb_types.float64
        _type_check(_nb_types.Array(T, 1, "C"), {rest_args})
        dt = _np.float32 if T == _nb_types.float32 else _np.float64

        def impl({args_with_types}):
            _arr = _np.empty(1, dt)
            _arr[0] = {first}
            return {impl}(_arr, {rest_args})[0]

        return impl
    _type_check({args})
    return {impl}.__wrapped__
"""

        if fname == "integrate" and "_pmf" in d:
            returns = (
                "float\n    Probability of lo < k <= hi, which is cdf(hi) - cdf(lo)."
            )
        elif fname == "integrate":
            returns = "float\n    Integral of the density from lo to hi."
        else:
            returns = "Array-like\n    Function evaluated at the x points."
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
{returns}
\"\"\"
"""
        exec(code, d)
