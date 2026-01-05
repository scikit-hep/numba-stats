import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import norm


def test_pdf_one():
    x = 1
    got = norm.pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    assert_allclose(got, expected)


def test_pdf():
    x = np.linspace(-5, 5, 10)
    got = norm.pdf(x, 1, 2)
    expected = sc.norm.pdf(x, 1, 2)
    assert_allclose(got, expected)


def test_logpdf():
    x = np.linspace(-5, 5, 10)
    got = norm.logpdf(x, 1, 2)
    expected = sc.norm.logpdf(x, 1, 2)
    assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-5, 5, 10)
    got = norm.cdf(x, 1, 2)
    expected = sc.norm.cdf(x, 1, 2)
    assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 10)
    got = norm.ppf(p, 0, 1)
    expected = sc.norm.ppf(p)
    assert_allclose(got, expected)

    got = norm.ppf(0.5, 0, 1)
    expected = sc.norm.ppf(0.5, 0, 1)
    assert_allclose(got, expected)


def test_rvs():
    mu = 2
    sigma = 3
    x = norm.rvs(mu, sigma, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: norm.cdf(x, mu, sigma))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [norm.logpdf, norm.pdf, norm.cdf, norm.ppf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 0.0, 1.0)

    x = np.linspace(-3, 3, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 0, 1))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return norm.rvs(0.0, 1.0, 10, 1)

    assert_allclose(test(), norm.rvs(0, 1, 10, 1))
