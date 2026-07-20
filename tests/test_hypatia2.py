import numba as nb
import numpy as np
import pytest
import scipy.special as sp
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import hypatia2

# lambd, zeta, beta, a_left, n_left, a_right, n_right, loc, scale
pars = [
    (-1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0),
    (-2.5, 0.0, 0.3, 1.5, 3.0, 2.0, 1.5, 0.5, 1.2),
    (-1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0),
    (1.5, 3.0, -0.4, 1.2, 2.5, 0.8, 4.0, -1.0, 2.0),
    (-0.5, 0.5, 0.1, 2.0, 1.5, 2.0, 1.5, 0.0, 1.0),
]

# only these can be integrated analytically
pars_integrable = [
    (-1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0),
    (-2.5, 0.0, 0.0, 1.5, 3.0, 2.0, 1.5, 0.5, 1.2),
    (-0.3, 0.0, 0.0, 0.5, 4.0, 3.0, 2.5, -1.0, 2.0),
    # the configuration used by ROOT's own test of the analytical integral
    (-1.0, 0.0, 0.0, 50.0, 1.5, 1.0, 0.1, 0.0, 1.0),
]

# values of RooHypatia2 obtained with ROOT 6.40.02, computed with getVal without a
# normalization set, which returns the unnormalized density
root_x = [-6.0, -1.5, 0.0, 1.5, 6.0]
root_density = [
    (
        (-1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0),
        [
            0.015669956369784986,
            0.18700344626421092,
            1.0,
            0.18700344626421092,
            0.015669956369784986,
        ],
    ),
    (
        (-2.5, 0.0, 0.3, 1.5, 3.0, 2.0, 1.5, 0.5, 1.2),
        [
            0.00012908015620028334,
            0.010495360315280481,
            0.5324553439503407,
            0.27746380765908163,
            0.0017140989919808235,
        ],
    ),
    (
        (1.5, 3.0, -0.4, 1.2, 2.5, 0.8, 4.0, -1.0, 2.0),
        [
            0.11537856361852265,
            0.25708252918162655,
            0.12306878151860472,
            0.03475169945675022,
            0.0028870825724881546,
        ],
    ),
]

# integrals of the density over (loc - 7, loc + 7), obtained with
# RooHypatia2::createIntegral of ROOT 6.40.02
root_integral = [
    ((-1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0), 2.1856027782129654),
    ((-2.5, 0.0, 0.0, 1.5, 3.0, 2.0, 1.5, 0.5, 1.2), 1.4204544860630102),
    ((-1.0, 0.0, 0.0, 50.0, 1.5, 1.0, 0.1, 0.0, 1.0), 3.1888207159955715),
]

logsq2pi = np.log(np.sqrt(2 * np.pi))


def roofit_bessel_k(ni, x, log=False):
    # transcription of besselK and LnBesselK of RooHypatia2
    nu = abs(ni)
    if (x < 1e-6 and nu > 0) or (x < 1e-4 and 0 < nu < 55) or (x < 0.1 and nu >= 55):
        r = sp.gamma(nu) * 2.0 ** (nu - 1.0) * x**-nu
    else:
        r = sp.kv(nu, x)
    return np.log(r) if log else r


def roofit_log_eval(d, lambd, alpha, beta, delta):
    # transcription of LogEval of RooHypatia2
    thing = delta * delta + d * d
    logno = (
        lambd * np.log(alpha / delta)
        - logsq2pi
        - roofit_bessel_k(lambd, delta * alpha, log=True)
    )
    return np.exp(
        logno
        + beta * d
        + (0.5 - lambd) * (np.log(alpha) - 0.5 * np.log(thing))
        + roofit_bessel_k(lambd - 0.5, alpha * np.sqrt(thing), log=True)
    )


def roofit_diff_eval(d, lambd, alpha, beta, delta):
    # transcription of diff_eval of RooHypatia2
    thing = delta * delta + d * d
    alphasq = alpha * np.sqrt(thing)
    no = (alpha / delta) ** lambd / roofit_bessel_k(lambd, delta * alpha)
    ns1 = 0.5 - lambd
    return (
        no
        / np.sqrt(2 * np.pi)
        * alpha**ns1
        * thing ** (lambd / 2.0 - 1.25)
        * (
            -d
            * alphasq
            * (
                roofit_bessel_k(lambd - 1.5, alphasq)
                + roofit_bessel_k(lambd + 0.5, alphasq)
            )
            + (2.0 * (beta * thing + d * lambd) - d) * roofit_bessel_k(ns1, alphasq)
        )
        * np.exp(beta * d)
        * 0.5
    )


def roofit_density(x, lambd, zeta, beta, a, n, a2, n2, mu, sigma):
    # transcription of RooHypatia2::evaluate, which is not normalized to unity
    d = x - mu
    asigma = a * sigma
    a2sigma = a2 * sigma
    if zeta > 0.0:
        phi = roofit_bessel_k(lambd + 1.0, zeta) / roofit_bessel_k(lambd, zeta)
        cons1 = sigma / np.sqrt(phi)
        alpha = np.sqrt(zeta) / cons1
        delta = np.sqrt(zeta) * cons1
        if d < -asigma:
            k1 = roofit_log_eval(-asigma, lambd, alpha, beta, delta)
            k2 = roofit_diff_eval(-asigma, lambd, alpha, beta, delta)
            b = -asigma + n * k1 / k2
            return k1 * (b + asigma) ** n * (b - d) ** -n
        if d > a2sigma:
            k1 = roofit_log_eval(a2sigma, lambd, alpha, beta, delta)
            k2 = roofit_diff_eval(a2sigma, lambd, alpha, beta, delta)
            b = -a2sigma - n2 * k1 / k2
            return k1 * (b + a2sigma) ** n2 * (b + d) ** -n2
        return roofit_log_eval(d, lambd, alpha, beta, delta)
    delta = sigma
    if d < -asigma:
        cons1 = np.exp(-beta * asigma)
        phi = 1.0 + a * a
        k1 = cons1 * phi ** (lambd - 0.5)
        k2 = beta * k1 - cons1 * (lambd - 0.5) * phi ** (lambd - 1.5) * 2.0 * a / delta
        b = -asigma + n * k1 / k2
        return k1 * (b + asigma) ** n * (b - d) ** -n
    if d > a2sigma:
        cons1 = np.exp(beta * a2sigma)
        phi = 1.0 + a2 * a2
        k1 = cons1 * phi ** (lambd - 0.5)
        k2 = beta * k1 + cons1 * (lambd - 0.5) * phi ** (lambd - 1.5) * 2.0 * a2 / delta
        b = -a2sigma - n2 * k1 / k2
        return k1 * (b + a2sigma) ** n2 * (b + d) ** -n2
    return np.exp(beta * d) * (1.0 + d * d / (delta * delta)) ** (lambd - 0.5)


@pytest.mark.parametrize("par", pars)
def test_density(par):
    x = np.linspace(-10, 10, 501)
    got = hypatia2.density(x, *par)
    lambd, zeta, beta, a_left, n_left, a_right, n_right, loc, scale = par
    expected = [
        roofit_density(
            xi, lambd, zeta, beta, a_left, n_left, a_right, n_right, loc, scale
        )
        for xi in x
    ]
    assert_allclose(got, expected, rtol=1e-13)


@pytest.mark.parametrize(("par", "expected"), root_density)
def test_density_vs_root(par, expected):
    assert_allclose(hypatia2.density(root_x, *par), expected, rtol=1e-13)


@pytest.mark.parametrize(("par", "expected"), root_integral)
def test_integral_vs_root(par, expected):
    loc = par[-2]
    got = hypatia2.integral(loc + 7, *par) - hypatia2.integral(loc - 7, *par)
    assert got == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("par", pars)
def test_density_is_smooth(par):
    # the core and the tails are joined so that the density and its first
    # derivative are continuous
    loc = par[-2]
    scale = par[-1]
    h = 1e-7
    for edge in (loc - par[3] * scale, loc + par[5] * scale):
        below = hypatia2.density([edge - 2 * h, edge - h], *par)
        above = hypatia2.density([edge + h, edge + 2 * h], *par)
        assert_allclose(above[0], below[1], rtol=1e-5)
        assert_allclose(np.diff(above), np.diff(below), rtol=1e-4)


@pytest.mark.parametrize("zeta", (1e-6, 1e-8))
def test_density_vs_zeta_zero(zeta):
    # the zeta > 0 branch approaches the zeta == 0 branch, up to a constant factor
    x = np.linspace(-6, 6, 101)
    par = (-1.5, 0.0, 0.0, 1.5, 3.0, 2.0, 2.5, 0.0, 1.0)
    expected = hypatia2.density(x, *par)
    got = hypatia2.density(x, par[0], zeta, *par[2:])
    assert_allclose(got / got.max(), expected / expected.max(), rtol=1e-10)


@pytest.mark.parametrize("par", pars_integrable)
def test_integral(par):
    loc = par[-2]
    for lo, hi in ((loc - 7, loc + 7), (loc - 0.5, loc + 0.5), (loc + 2, loc + 30)):
        expected = quad(lambda x: hypatia2.density(x, *par), lo, hi, limit=400)[0]
        got = hypatia2.integral(hi, *par) - hypatia2.integral(lo, *par)
        assert got == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("par", pars_integrable)
def test_integral_at_loc(par):
    assert hypatia2.integral(par[-2], *par) == 0


def test_density_bad_parameters():
    x = np.linspace(-1, 1, 10)
    with pytest.raises(ValueError):
        hypatia2.density(x, -1.0, -1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        hypatia2.density(x, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0)


def test_integral_bad_parameters():
    x = np.linspace(-1, 1, 10)
    # analytical integration requires zeta == 0, beta == 0, and lambd < 0
    with pytest.raises(ValueError):
        hypatia2.integral(x, -1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        hypatia2.integral(x, -1.0, 0.0, 0.2, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        hypatia2.integral(x, 1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [hypatia2.density, hypatia2.integral])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, -1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0)

    x = np.linspace(-5, 5, 1000)
    y = test(x)

    assert_allclose(y, fn(x, -1, 0, 0, 1, 2, 1, 2, 0, 1))
