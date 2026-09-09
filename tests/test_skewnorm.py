import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import norm, skewnorm

As = (-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0)


def test_pdf_one():
    got = skewnorm.pdf(1, 3, 1, 2)
    expected = sc.skewnorm.pdf(1, 3, 1, 2)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", As)
def test_pdf(a):
    x = np.linspace(-10, 10, 50)
    got = skewnorm.pdf(x, a, 1, 2)
    expected = sc.skewnorm.pdf(x, a, 1, 2)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", As)
def test_logpdf(a):
    # includes the tail opposite to the skew, where the pdf underflows
    x = np.linspace(-30, 30, 50)
    got = skewnorm.logpdf(x, a, 1, 2)
    expected = sc.skewnorm.logpdf(x, a, 1, 2)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", (-3.0, 0.0, 3.0))
def test_pdf_is_normalized(a):
    got = quad(lambda x: skewnorm.pdf(x, a, 1, 2), -np.inf, np.inf)[0]
    assert_allclose(got, 1)


@pytest.mark.parametrize("a", As)
def test_cdf(a):
    x = np.linspace(-10, 10, 50)
    got = skewnorm.cdf(x, a, 1, 2)
    expected = sc.skewnorm.cdf(x, a, 1, 2)
    # absolute precision of Owen's T function is limited to about 1e-16
    assert_allclose(got, expected, atol=1e-14)


def test_vs_norm():
    x = np.linspace(-10, 10, 50)
    assert_allclose(skewnorm.pdf(x, 0, 1, 2), norm.pdf(x, 1, 2))
    assert_allclose(skewnorm.logpdf(x, 0, 1, 2), norm.logpdf(x, 1, 2))
    assert_allclose(skewnorm.cdf(x, 0, 1, 2), norm.cdf(x, 1, 2))


@pytest.mark.parametrize("a", (-3.0, 0.0, 3.0))
def test_rvs(a):
    x = skewnorm.rvs(a, 2, 3, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: skewnorm.cdf(x, a, 2, 3))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [skewnorm.logpdf, skewnorm.pdf, skewnorm.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 2.0, 0.0, 1.0)

    x = np.linspace(-5, 5, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 2, 0, 1))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return skewnorm.rvs(2.0, 0.0, 1.0, 10, 1)

    assert_allclose(test(), skewnorm.rvs(2, 0, 1, 10, 1))
