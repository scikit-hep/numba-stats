import numpy as np
import pytest
import scipy.stats as sc
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
