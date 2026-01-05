import numpy as np
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
