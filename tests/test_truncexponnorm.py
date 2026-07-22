import numpy as np
import pytest
import scipy.stats as sc
from scipy.integrate import quad

from numba_stats import truncexponnorm


@pytest.mark.parametrize("mu", (0, -1, 1))
@pytest.mark.parametrize("sigma", (1, 0.5, 2))
@pytest.mark.parametrize("tau", (1, 2, 3))
def test_truncnorm(mu, sigma, tau):
    got = quad(lambda x: truncexponnorm.pdf(x, -1, 1, mu, sigma, tau), -10, 10)[0]
    expected = 1.0
    np.testing.assert_allclose(got, expected)

def test_pdf():
    x = np.linspace(-1, 5, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    tau = 1
    K = tau / sigma
    got = truncexponnorm.pdf(x, xmin, xmax, mu, sigma, tau)
    # We have to manually truncate the distribution since scipy doesn't provide one
    pmin = sc.exponnorm.cdf(xmin, K, loc=mu, scale=sigma)
    pmax = sc.exponnorm.cdf(xmax, K, loc=mu, scale=sigma)
    expected = np.where((xmin <= x) & (x < xmax), sc.exponnorm.pdf(x, K, loc=mu, scale=sigma) / (pmax - pmin), 0.0)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-1, 5, 10)
    xmin = 1
    xmax = 4
    mu = 2
    sigma = 3
    tau = 1
    K = tau / sigma
    got = truncexponnorm.cdf(x, xmin, xmax, mu, sigma, tau)
    # We have to manually truncate the distribution since scipy doesn't provide one
    pmin = sc.exponnorm.cdf(xmin, K, loc=mu, scale=sigma)
    pmax = sc.exponnorm.cdf(xmax, K, loc=mu, scale=sigma)
    expected = np.where(x < xmin, 0.0, np.where(x > xmax, 1.0, (sc.exponnorm.cdf(x, K, loc=mu, scale=sigma) - pmin) / (pmax - pmin)))
    np.testing.assert_allclose(got, expected)