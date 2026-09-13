import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import truncnorm


@pytest.mark.parametrize("mu", (0, -1, 1))
@pytest.mark.parametrize("sigma", (1, 0.5, 2))
def test_truncnorm(mu, sigma):
    got = quad(lambda x: truncnorm.pdf(x, -1, 1, mu, sigma), -10, 10)[0]
    expected = 1.0
    np.testing.assert_allclose(got, expected)


def test_logpdf():
    x = np.linspace(-1, 5, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    got = truncnorm.logpdf(x, xmin, xmax, mu, sigma)
    z = (x - mu) / sigma
    zmin = (xmin - mu) / sigma
    zmax = (xmax - mu) / sigma
    expected = sc.truncnorm.logpdf(z, zmin, zmax) - np.log(sigma)
    np.testing.assert_allclose(got, expected)


def test_pdf():
    x = np.linspace(-1, 5, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    got = truncnorm.pdf(x, xmin, xmax, mu, sigma)
    z = (x - mu) / sigma
    zmin = (xmin - mu) / sigma
    zmax = (xmax - mu) / sigma
    expected = sc.truncnorm.pdf(z, zmin, zmax) / sigma
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-1, 5, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    got = truncnorm.cdf(x, xmin, xmax, mu, sigma)
    z = (x - mu) / sigma
    zmin = (xmin - mu) / sigma
    zmax = (xmax - mu) / sigma
    expected = sc.truncnorm.cdf(z, zmin, zmax)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    expected = np.linspace(0, 1, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    x = truncnorm.ppf(expected, mu, sigma, xmin, xmax)
    got = truncnorm.cdf(x, mu, sigma, xmin, xmax)
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_rvs():
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    x = truncnorm.rvs(xmin, xmax, mu, sigma, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: truncnorm.cdf(x, xmin, xmax, mu, sigma))
    assert r.pvalue > 0.01


def test_integrate():
    par = -1, 3, 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = truncnorm.integrate(lo, hi, *par)
        expected = np.diff(truncnorm.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(truncnorm.integrate(-1, 3, *par), 1)
    assert_allclose(truncnorm.integrate(-10, 10, *par), 1)


def test_integrate_tail():
    got = truncnorm.integrate(7, 8, -1, 9, 0, 1)
    expected = (sc.norm.sf(7) - sc.norm.sf(8)) / (sc.norm.cdf(9) - sc.norm.cdf(-1))
    assert_allclose(got, expected, rtol=1e-10)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return truncnorm.integrate(lo, hi, -1.0, 3.0, 1.0, 2.0)

    expected = truncnorm.integrate(-1.0, 1.5, -1.0, 3.0, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
