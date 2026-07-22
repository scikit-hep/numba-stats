import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import exponnorm


def test_pdf_one():
    x = 1
    mu = 1
    sigma = 2
    tau = 3
    K = tau / sigma
    got = exponnorm.pdf(x, mu, sigma, tau)
    expected = sc.exponnorm.pdf(x, K, loc=mu, scale=sigma)
    assert_allclose(got, expected)


def test_pdf():
    mu = 1
    sigma = 2
    tau = 3
    K = tau / sigma
    x = np.linspace(-5, 5, 10)
    got = exponnorm.pdf(x, mu, sigma, tau)
    expected = sc.exponnorm.pdf(x, K, loc=mu, scale=sigma)
    assert_allclose(got, expected)


def test_cdf_one():
    x = 1
    mu = 1
    sigma = 2
    tau = 3
    K = tau / sigma
    got = exponnorm.cdf(x, mu, sigma, tau)
    expected = sc.exponnorm.cdf(x, K, loc=mu, scale=sigma)
    assert_allclose(got, expected)


def test_cdf():
    mu = 1
    sigma = 2
    tau = 3
    K = tau / sigma
    x = np.linspace(-5, 5, 10)
    got = exponnorm.cdf(x, mu, sigma, tau)
    expected = sc.exponnorm.cdf(x, K, loc=mu, scale=sigma)
    assert_allclose(got, expected)


def test_rvs():
    mu = 2
    sigma = 3
    tau = 2
    x = exponnorm.rvs(mu, sigma, tau, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: exponnorm.cdf(x, mu, sigma, tau))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [exponnorm.pdf, exponnorm.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 0.0, 1.0, 2.0)

    x = np.linspace(-3, 3, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 0, 1, 2))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return exponnorm.rvs(0.0, 1.0, 2.0, 10, 1)

    assert_allclose(test(), exponnorm.rvs(0, 1, 2, 10, 1))
