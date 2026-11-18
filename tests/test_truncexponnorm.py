import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import truncexponnorm


@pytest.mark.parametrize("K", (0.03, 0.5, 3.0))
@pytest.mark.parametrize("mu", (0, -1, 1))
@pytest.mark.parametrize("sigma", (1, 0.5, 2))
def test_pdf_is_normalized(K, mu, sigma):
    got = quad(lambda x: truncexponnorm.pdf(x, -1, 1, K, mu, sigma), -1, 1)[0]
    assert_allclose(got, 1)


def scipy_pdf(x, xmin, xmax, K, mu, sigma):
    # scipy has no truncated version, we truncate manually
    pmin = sc.exponnorm.cdf(xmin, K, mu, sigma)
    pmax = sc.exponnorm.cdf(xmax, K, mu, sigma)
    return np.where(
        (xmin <= x) & (x < xmax),
        sc.exponnorm.pdf(x, K, mu, sigma) / (pmax - pmin),
        0.0,
    )


def scipy_cdf(x, xmin, xmax, K, mu, sigma):
    pmin = sc.exponnorm.cdf(xmin, K, mu, sigma)
    pmax = sc.exponnorm.cdf(xmax, K, mu, sigma)
    return np.clip((sc.exponnorm.cdf(x, K, mu, sigma) - pmin) / (pmax - pmin), 0, 1)


@pytest.mark.parametrize("K", (0.03, 0.5, 3.0))
def test_pdf(K):
    x = np.linspace(-1, 5, 10)
    par = 1, 4, K, 2, 3
    got = truncexponnorm.pdf(x, *par)
    expected = scipy_pdf(x, *par)
    assert_allclose(got, expected)


@pytest.mark.parametrize("K", (0.03, 0.5, 3.0))
def test_logpdf(K):
    x = np.linspace(-1, 5, 10)
    par = 1, 4, K, 2, 3
    got = truncexponnorm.logpdf(x, *par)
    with np.errstate(divide="ignore"):
        expected = np.log(scipy_pdf(x, *par))
    assert_allclose(got, expected)


@pytest.mark.parametrize("K", (0.03, 0.5, 3.0))
def test_cdf(K):
    x = np.linspace(-1, 5, 10)
    par = 1, 4, K, 2, 3
    got = truncexponnorm.cdf(x, *par)
    expected = scipy_cdf(x, *par)
    assert_allclose(got, expected)


def test_support():
    par = 1, 4, 0.5, 2, 3
    assert truncexponnorm.pdf(1, *par) > 0
    assert truncexponnorm.pdf(4, *par) == 0
    assert truncexponnorm.logpdf(0, *par) == -np.inf
    assert truncexponnorm.cdf(0, *par) == 0
    assert truncexponnorm.cdf(1, *par) == 0
    assert truncexponnorm.cdf(4, *par) == 1


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "fn", [truncexponnorm.logpdf, truncexponnorm.pdf, truncexponnorm.cdf]
)
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, -1.0, 4.0, 0.5, 1.0, 2.0)

    x = np.linspace(-3, 6, 1000)
    y = test(x)

    assert_allclose(y, fn(x, -1, 4, 0.5, 1, 2))


def test_integrate():
    par = -1, 4, 1.5, 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = truncexponnorm.integrate(lo, hi, *par)
        expected = np.diff(truncexponnorm.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(truncexponnorm.integrate(-1, 4, *par), 1)
    assert_allclose(truncexponnorm.integrate(-10, 10, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return truncexponnorm.integrate(lo, hi, -1.0, 4.0, 1.5, 1.0, 2.0)

    expected = truncexponnorm.integrate(-1.0, 1.5, -1.0, 4.0, 1.5, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
