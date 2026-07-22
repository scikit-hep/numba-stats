import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import skewnorm

def test_pdf_one():
    x = 1
    got = skewnorm.pdf(x, 1, 2, 3)
    expected = sc.skewnorm.pdf(x, 3, loc=1, scale=2)
    assert_allclose(got, expected, atol=1e-16)


def test_pdf():
    x = np.linspace(-5, 5, 10)
    got = skewnorm.pdf(x, 1, 2, 3)
    expected = sc.skewnorm.pdf(x, 3, loc=1, scale=2)
    assert_allclose(got, expected, atol=1e-16)

def test_cdf():
    x = np.linspace(-5, 5, 10)
    got = skewnorm.cdf(x, 1, 2, 3)
    expected = sc.skewnorm.cdf(x, 3, loc=1, scale=2)
    assert_allclose(got, expected, atol=1e-16)

def test_rvs():
    mu = 2
    sigma = 3
    a = 2
    x = skewnorm.rvs(mu, sigma, a, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: skewnorm.cdf(x, mu, sigma, a))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [skewnorm.pdf, skewnorm.cdf])
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
        return skewnorm.rvs(0.0, 1.0, 2.0, 10, 1)

    assert_allclose(test(), skewnorm.rvs(0, 1, 2, 10, 1))