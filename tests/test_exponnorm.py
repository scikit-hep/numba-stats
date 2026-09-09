import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import exponnorm, norm

# includes values where the textbook formula overflows (K < 0.03)
Ks = (0.01, 0.03, 0.1, 1.0, 10.0, 100.0)


def test_pdf_one():
    got = exponnorm.pdf(1, 1.5, 1, 2)
    expected = sc.exponnorm.pdf(1, 1.5, 1, 2)
    assert_allclose(got, expected)


@pytest.mark.parametrize("K", Ks)
def test_pdf(K):
    x = np.linspace(-10, 30, 50)
    got = exponnorm.pdf(x, K, 1, 2)
    expected = sc.exponnorm.pdf(x, K, 1, 2)
    assert_allclose(got, expected)


@pytest.mark.parametrize("K", Ks)
def test_logpdf(K):
    x = np.linspace(-10, 30, 50)
    got = exponnorm.logpdf(x, K, 1, 2)
    expected = sc.exponnorm.logpdf(x, K, 1, 2)
    assert_allclose(got, expected)


@pytest.mark.parametrize("K", (0.1, 1.0, 10.0))
def test_pdf_is_normalized(K):
    got = quad(lambda x: exponnorm.pdf(x, K, 1, 2), -np.inf, np.inf)[0]
    assert_allclose(got, 1)


@pytest.mark.parametrize("K", Ks)
def test_cdf(K):
    x = np.linspace(-10, 30, 50)
    got = exponnorm.cdf(x, K, 1, 2)
    expected = sc.exponnorm.cdf(x, K, 1, 2)
    assert_allclose(got, expected)


def test_vs_norm():
    # the distribution becomes a normal distribution for K -> 0
    x = np.linspace(-5, 5, 20)
    assert_allclose(exponnorm.pdf(x, 1e-4, 1, 2), norm.pdf(x, 1, 2), rtol=1e-3)
    assert_allclose(exponnorm.cdf(x, 1e-4, 1, 2), norm.cdf(x, 1, 2), rtol=1e-3)


@pytest.mark.parametrize("K", (0.1, 1.0, 10.0))
def test_rvs(K):
    x = exponnorm.rvs(K, 2, 3, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: exponnorm.cdf(x, K, 2, 3))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [exponnorm.logpdf, exponnorm.pdf, exponnorm.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 0.5, 1.0, 2.0)

    x = np.linspace(-5, 15, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 0.5, 1, 2))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return exponnorm.rvs(0.5, 1.0, 2.0, 10, 1)

    assert_allclose(test(), exponnorm.rvs(0.5, 1, 2, 10, 1))
