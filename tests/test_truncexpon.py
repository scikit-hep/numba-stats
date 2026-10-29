import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import kstest

from numba_stats import expon, truncexpon


def test_pdf():
    x = np.linspace(-5, 5, 100)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    got = truncexpon.pdf(x, xmin, xmax, mu, sigma)
    expected = expon.pdf(x, mu, sigma) / (
        expon.cdf(xmax, mu, sigma) - expon.cdf(xmin, mu, sigma)
    )
    expected[x < xmin] = 0
    expected[x > xmax] = 0
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-5, 5, 100)
    xmin = 1
    xmax = 4
    mu = 1.5
    sigma = 2
    got = truncexpon.cdf(x, xmin, xmax, mu, sigma)
    expected = (expon.cdf(x, mu, sigma) - expon.cdf(xmin, mu, sigma)) / (
        expon.cdf(xmax, mu, sigma) - expon.cdf(xmin, mu, sigma)
    )
    expected[x < xmin] = 0
    expected[x > xmax] = 1
    np.testing.assert_allclose(got, expected)


def test_ppf():
    expected = np.linspace(0, 1, 100)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    x = truncexpon.ppf(expected, xmin, xmax, mu, sigma)
    got = truncexpon.cdf(x, xmin, xmax, mu, sigma)
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_rvs():
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    x = truncexpon.rvs(xmin, xmax, mu, sigma, size=100_000, random_state=1)
    r = kstest(x, lambda x: truncexpon.cdf(x, xmin, xmax, mu, sigma))
    assert r.pvalue > 0.01


def test_integrate():
    par = 0.5, 3, 0, 1
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = truncexpon.integrate(lo, hi, *par)
        expected = np.diff(truncexpon.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(truncexpon.integrate(0.5, 3, *par), 1)
    assert_allclose(truncexpon.integrate(-10, 10, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return truncexpon.integrate(lo, hi, 0.5, 3.0, 0.0, 1.0)

    expected = truncexpon.integrate(1.0, 2.0, 0.5, 3.0, 0.0, 1.0)
    assert_allclose(test(1.0, 2.0), expected)
