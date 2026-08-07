"""
Hypatia2 distribution.

The two-sided Hypatia distribution, see https://arxiv.org/abs/1312.5000. It consists of
a hyperbolic core, which is a generalised hyperbolic distribution, and two power-law
tails, which are attached a_left and a_right scale units away from the mode. The
function and its first derivative are continuous at the junctions.

The density is not normalized to unity, use this in extended likelihood fits. There is
no closed form for the integral of the generalised hyperbolic core, but the special case
zeta = 0, beta = 0, and lambd < 0 can be integrated analytically, which is provided by
the function integral. It returns the integral of the density from loc to x.

Notes
-----
This implementation was modeled after
https://root.cern.ch/doc/master/RooHypatia2_8cxx_source.html. The parameters "mu",
"sigma", "a", "n", "a2", and "n2" of the original implementation are called "loc",
"scale", "a_left", "n_left", "a_right", and "n_right" here to follow the conventions of
this library.
"""

from math import gamma as _gamma
from math import lgamma as _lgamma

import numpy as np

from ._special import hyp2f1 as _hyp2f1
from ._special import kv as _kv
from ._util import _generate_wrappers, _jit, _jit_pointwise, _prange

_doc_par = """
lambd : float
    Shape parameter of the core. Must be negative if zeta is zero.
zeta : float
    Shape parameter of the core. Must be non-negative.
beta : float
    Asymmetry parameter of the core. The symmetric case is beta = 0, values should be
    chosen close to zero.
a_left : float
    Distance from the mode in units of scale where the core turns into a power law on
    the left-hand side. Must be non-negative.
n_left : float
    Shape parameter of the left power-law tail. Must be non-negative.
a_right : float
    Distance from the mode in units of scale where the core turns into a power law on
    the right-hand side. Must be non-negative.
n_right : float
    Shape parameter of the right power-law tail. Must be non-negative.
loc : float
    Location of the mode of the distribution.
scale : float
    Width parameter.
"""


@_jit_pointwise(2, cache=False)  # cannot cache because of _kv
def _bessel_k(nu: float, x: float) -> float:
    T = type(nu)
    nu = abs(nu)
    if (x < 1e-6 and nu > 0) or (x < 1e-4 and 0 < nu < 55) or (x < 0.1 and nu >= 55):
        # small argument expansion, where the Bessel function is hard to compute
        return (  # type:ignore[no-any-return]
            T(_gamma(nu)) * T(2) ** (nu - T(1)) * x**-nu
        )
    return T(_kv(nu, x))


@_jit_pointwise(2, cache=False)  # cannot cache because of _kv
def _log_bessel_k(nu: float, x: float) -> float:
    T = type(nu)
    nu = abs(nu)
    if (x < 1e-6 and nu > 0) or (x < 1e-4 and 0 < nu < 55) or (x < 0.1 and nu >= 55):
        return (  # type:ignore[no-any-return]
            T(_lgamma(nu)) + T(np.log(2)) * (nu - T(1)) - np.log(x) * nu
        )
    return np.log(T(_kv(nu, x)))  # type:ignore[no-any-return]


@_jit_pointwise(5, cache=False)
def _core(d: float, lambd: float, alpha: float, beta: float, delta: float) -> float:
    T = type(d)
    half = T(0.5)
    thing = delta * delta + d * d
    logno = (
        lambd * np.log(alpha / delta)
        - T(0.5 * np.log(2 * np.pi))
        - _log_bessel_k(lambd, delta * alpha)
    )
    return np.exp(  # type:ignore[no-any-return]
        logno
        + beta * d
        + (half - lambd) * (np.log(alpha) - half * np.log(thing))
        + _log_bessel_k(lambd - half, alpha * np.sqrt(thing))
    )


@_jit_pointwise(5, cache=False)
def _core_diff(
    d: float, lambd: float, alpha: float, beta: float, delta: float
) -> float:
    T = type(d)
    half = T(0.5)
    two = T(2)
    thing = delta * delta + d * d
    alphasq = alpha * np.sqrt(thing)
    no = (
        (alpha / delta) ** lambd
        / _bessel_k(lambd, delta * alpha)
        * T(1 / np.sqrt(2 * np.pi))
    )
    ns1 = half - lambd
    return (  # type:ignore[no-any-return]
        no
        * alpha**ns1
        * thing ** (lambd / two - T(1.25))
        * (
            -d
            * alphasq
            * (_bessel_k(lambd - T(1.5), alphasq) + _bessel_k(lambd + half, alphasq))
            + (two * (beta * thing + d * lambd) - d) * _bessel_k(ns1, alphasq)
        )
        * np.exp(beta * d)
        * half
    )


@_jit_pointwise(3, cache=False)  # cannot cache because of _hyp2f1
def _core_integral(d: float, lambd: float, delta: float) -> float:
    # integral of (1 + d ** 2 / delta ** 2) ** (lambd - 0.5) from 0 to d
    T = type(d)
    return d * T(_hyp2f1(T(0.5), T(0.5) - lambd, T(1.5), -d * d / (delta * delta)))


@_jit(9, cache=False)
def _density(
    x: np.ndarray,
    lambd: float,
    zeta: float,
    beta: float,
    a_left: float,
    n_left: float,
    a_right: float,
    n_right: float,
    loc: float,
    scale: float,
) -> np.ndarray:
    if zeta < 0:
        raise ValueError("zeta < 0 is not supported")
    if zeta == 0 and lambd >= 0:
        raise ValueError("lambd >= 0 is only supported for zeta > 0")

    T = type(lambd)
    half = T(0.5)
    two = T(2)
    s_left = a_left * scale
    s_right = a_right * scale

    # the tails are k_left * (b_left - d) ** -n_left and
    # k_right * (b_right + d) ** -n_right
    if zeta > 0:
        phi = _bessel_k(lambd + T(1), zeta) / _bessel_k(lambd, zeta)
        cons = scale / np.sqrt(phi)
        alpha = np.sqrt(zeta) / cons
        delta = np.sqrt(zeta) * cons

        k1 = _core(-s_left, lambd, alpha, beta, delta)
        k2 = _core_diff(-s_left, lambd, alpha, beta, delta)
        b_left = -s_left + n_left * k1 / k2
        k_left = k1 * (b_left + s_left) ** n_left

        k1 = _core(s_right, lambd, alpha, beta, delta)
        k2 = _core_diff(s_right, lambd, alpha, beta, delta)
        b_right = -s_right - n_right * k1 / k2
        k_right = k1 * (b_right + s_right) ** n_right
    else:
        alpha = T(0)  # not used, the core is computed in closed form
        delta = scale

        e = np.exp(-beta * s_left)
        phi = T(1) + a_left * a_left
        k1 = e * phi ** (lambd - half)
        k2 = (
            beta * k1
            - e * (lambd - half) * phi ** (lambd - T(1.5)) * two * a_left / delta
        )
        b_left = -s_left + n_left * k1 / k2
        k_left = k1 * (b_left + s_left) ** n_left

        e = np.exp(beta * s_right)
        phi = T(1) + a_right * a_right
        k1 = e * phi ** (lambd - half)
        k2 = (
            beta * k1
            + e * (lambd - half) * phi ** (lambd - T(1.5)) * two * a_right / delta
        )
        b_right = -s_right - n_right * k1 / k2
        k_right = k1 * (b_right + s_right) ** n_right

    r = np.empty_like(x)
    for i in _prange(len(r)):
        d = x[i] - loc
        if d < -s_left:
            r[i] = k_left * (b_left - d) ** -n_left
        elif d > s_right:
            r[i] = k_right * (b_right + d) ** -n_right
        elif zeta > 0:
            r[i] = _core(d, lambd, alpha, beta, delta)
        else:
            r[i] = np.exp(beta * d) * (T(1) + d * d / (delta * delta)) ** (lambd - half)
    return r


@_jit(9, cache=False)
def _integral(
    x: np.ndarray,
    lambd: float,
    zeta: float,
    beta: float,
    a_left: float,
    n_left: float,
    a_right: float,
    n_right: float,
    loc: float,
    scale: float,
) -> np.ndarray:
    if zeta != 0 or beta != 0 or lambd >= 0:
        raise ValueError("integral requires zeta == 0, beta == 0, and lambd < 0")

    T = type(lambd)
    one = T(1)
    half = T(0.5)
    two = T(2)
    s_left = a_left * scale
    s_right = a_right * scale
    delta = scale

    phi = one + a_left * a_left
    k1 = phi ** (lambd - half)
    k2 = -(lambd - half) * phi ** (lambd - T(1.5)) * two * a_left / delta
    b_left = -s_left + n_left * k1 / k2
    k_left = k1 * (b_left + s_left) ** n_left

    phi = one + a_right * a_right
    k1 = phi ** (lambd - half)
    k2 = (lambd - half) * phi ** (lambd - T(1.5)) * two * a_right / delta
    b_right = -s_right - n_right * k1 / k2
    k_right = k1 * (b_right + s_right) ** n_right

    # offsets which make the antiderivative continuous at the junctions
    c_left = _core_integral(-s_left, lambd, delta) - k_left * (b_left + s_left) ** (
        one - n_left
    ) / (n_left - one)
    c_right = _core_integral(s_right, lambd, delta) - k_right * (b_right + s_right) ** (
        one - n_right
    ) / (one - n_right)

    r = np.empty_like(x)
    for i in _prange(len(r)):
        d = x[i] - loc
        if d < -s_left:
            r[i] = k_left * (b_left - d) ** (one - n_left) / (n_left - one) + c_left
        elif d > s_right:
            r[i] = (
                k_right * (b_right + d) ** (one - n_right) / (one - n_right) + c_right
            )
        else:
            r[i] = _core_integral(d, lambd, delta)
    return r


_generate_wrappers(globals())
