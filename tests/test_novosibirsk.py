import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import norm, novosibirsk

xi = 2.3548200450309494  # 2 sqrt(log(4))


def roofit_pdf(x, peak, width, tail):
    # transcription of RooNovosibirsk::evaluate, which is not normalized to unity
    if abs(tail) < 1e-7:
        return np.exp(-0.5 * ((x - peak) / width) ** 2)
    arg = 1.0 - (x - peak) * tail / width
    if arg < 1e-7:
        return 0.0
    width_zero = (2.0 / xi) * np.arcsinh(tail * xi * 0.5)
    width_zero2 = width_zero**2
    return np.exp(-0.5 / width_zero2 * np.log(arg) ** 2 - 0.5 * width_zero2)


def roofit_norm(width, tail):
    # integral of roofit_pdf over its support
    width_zero = (2.0 / xi) * np.arcsinh(tail * xi * 0.5)
    return np.sqrt(2 * np.pi) * abs(width_zero * width / tail)


@pytest.mark.parametrize("lambd", (-10.0, -3.0, -1.0, -0.1, 0.1, 1.0, 3.0, 10.0))
def test_pdf(lambd):
    peak = 90.0
    width = 10.0
    x = np.linspace(40, 140, 1000)
    got = novosibirsk.pdf(x, lambd, peak, width)
    expected = np.array([roofit_pdf(xj, peak, width, lambd) for xj in x])
    assert_allclose(got, expected / roofit_norm(width, lambd), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("lambd", (-3.0, -1.0, -0.1, 0.1, 1.0, 3.0))
def test_pdf_is_normalized(lambd):
    par = lambd, 1.5, 2.0
    # the tail is long, integrate over the bulk of the distribution
    a, b = novosibirsk.ppf([1e-12, 1 - 1e-12], *par)
    got = quad(lambda x: novosibirsk.pdf(x, *par), a, b, limit=200)[0]
    assert got == pytest.approx(1)


@pytest.mark.parametrize("lambd", (-3.0, -1.0, -0.1, 0.1, 1.0, 3.0))
def test_logpdf(lambd):
    par = lambd, 1.5, 2.0
    x = np.linspace(-10, 10, 20)
    with np.errstate(divide="ignore"):
        expected = np.log(novosibirsk.pdf(x, *par))
    assert_allclose(novosibirsk.logpdf(x, *par), expected)


@pytest.mark.parametrize("lambd", (-3.0, -1.0, -0.1, 0.1, 1.0, 3.0))
def test_cdf(lambd):
    loc = 1.5
    par = lambd, loc, 2.0
    x = np.linspace(-10, 10, 20)

    @np.vectorize
    def num_cdf(x):
        return quad(lambda x: novosibirsk.pdf(x, *par), loc, x, limit=200)[0]

    got = novosibirsk.cdf(x, *par) - novosibirsk.cdf(loc, *par)
    assert_allclose(got, num_cdf(x), atol=1e-9)


@pytest.mark.parametrize("lambd", (1e-10, 0.5, 1.0, 10.0, 20.0))
def test_cdf_vs_roofit(lambd):
    # integrals of the unnormalized RooNovosibirsk over the range 50 to 130. The
    # value for lambd = 20 was produced by RooAbsReal::createIntegral, the others
    # with the analytical formula of RooNovosibirsk::analyticalIntegral
    expected = {
        1e-10: 25.06469498570457,
        0.5: 23.021160811717586,
        1.0: 18.148056725398746,
        10.0: 0.6499205017648246,
        20.0: 0.11013822129162763,
    }[lambd]
    width = 10.0
    par = lambd, 90.0, width
    got = novosibirsk.cdf(130.0, *par) - novosibirsk.cdf(50.0, *par)
    assert got * roofit_norm(width, lambd) == pytest.approx(expected)


@pytest.mark.parametrize("lambd", (-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0))
def test_ppf(lambd):
    par = lambd, 1.5, 2.0
    p = np.linspace(0, 1, 20)
    assert_allclose(novosibirsk.cdf(novosibirsk.ppf(p, *par), *par), p)


@pytest.mark.parametrize("lambd", (-3.0, -0.5, 0.0, 0.5, 3.0))
def test_rvs(lambd):
    par = lambd, 1.5, 2.0
    x = novosibirsk.rvs(*par, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: novosibirsk.cdf(x, *par))
    assert r.pvalue > 0.01


@pytest.mark.parametrize("lambd", (0.0, 1e-10, -1e-10))
def test_vs_norm(lambd):
    # the distribution becomes a normal distribution for lambd -> 0
    x = np.linspace(-10, 10, 20)
    p = np.linspace(0.01, 0.99, 20)
    assert_allclose(novosibirsk.pdf(x, lambd, 1.5, 2.0), norm.pdf(x, 1.5, 2.0))
    assert_allclose(novosibirsk.cdf(x, lambd, 1.5, 2.0), norm.cdf(x, 1.5, 2.0))
    assert_allclose(novosibirsk.ppf(p, lambd, 1.5, 2.0), norm.ppf(p, 1.5, 2.0))


@pytest.mark.parametrize("lambd", (-2.0, 2.0))
def test_support(lambd):
    loc = 1.5
    scale = 2.0
    par = lambd, loc, scale
    edge = loc + scale / lambd
    inside = edge - 1e-6 * lambd
    outside = edge + 1e-6 * lambd

    assert novosibirsk.pdf(inside, *par) > 0
    assert novosibirsk.pdf(outside, *par) == 0
    assert novosibirsk.logpdf(outside, *par) == -np.inf
    assert novosibirsk.cdf(outside, *par) == (1 if lambd > 0 else 0)
    assert_allclose(novosibirsk.ppf(1 if lambd > 0 else 0, *par), edge)


@pytest.mark.parametrize("lambd", (-3.0, -0.5, 0.0, 0.5, 3.0))
def test_mode_and_fwhm(lambd):
    loc = 1.5
    scale = 2.0
    x = np.linspace(-20, 20, 200001)
    y = novosibirsk.pdf(x, lambd, loc, scale)
    # loc is the mode and scale is the standard deviation of the normal
    # distribution with the same full width at half maximum
    assert_allclose(x[np.argmax(y)], loc, atol=2e-4)
    (above,) = np.nonzero(y > 0.5 * np.max(y))
    fwhm = x[above[-1]] - x[above[0]]
    assert_allclose(fwhm, 2 * np.sqrt(2 * np.log(2)) * scale, atol=1e-3)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "fn", [novosibirsk.logpdf, novosibirsk.pdf, novosibirsk.cdf, novosibirsk.ppf]
)
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 0.5, 1.0, 2.0)

    x = np.linspace(0, 1, 1000) if fn is novosibirsk.ppf else np.linspace(-5, 5, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 0.5, 1, 2))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return novosibirsk.rvs(0.5, 1.0, 2.0, 10, 1)

    assert_allclose(test(), novosibirsk.rvs(0.5, 1, 2, 10, 1))
