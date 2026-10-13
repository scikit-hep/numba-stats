import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import gamma

As = (0.5, 1.0, 2.5, 10.0)

# includes points below and at the lower edge of the support, loc = 2
x = np.linspace(-1, 20, 22)


def test_pdf_one():
    got = gamma.pdf(3, 1, 2, 3)
    expected = sc.gamma.pdf(3, 1, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", As)
def test_pdf(a):
    got = gamma.pdf(x, a, 2, 3)
    expected = sc.gamma.pdf(x, a, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", As)
def test_logpdf(a):
    got = gamma.logpdf(x, a, 2, 3)
    expected = sc.gamma.logpdf(x, a, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", As)
def test_cdf(a):
    got = gamma.cdf(x, a, 2, 3)
    expected = sc.gamma.cdf(x, a, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", As)
def test_ppf(a):
    p = np.linspace(0, 1, 10)
    got = gamma.ppf(p, a, 2, 3)
    expected = sc.gamma.ppf(p, a, 2, 3)
    assert_allclose(got, expected)

    got = gamma.ppf(0.5, a, 2, 3)
    expected = sc.gamma.ppf(0.5, a, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("a", (0.5, 4.0))
def test_rvs(a):
    x = gamma.rvs(a, 2, 3, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: gamma.cdf(x, a, 2, 3))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [gamma.logpdf, gamma.pdf, gamma.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 3.0, 1.0, 2.0)

    x = np.linspace(1.1, 20, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 3, 1, 2))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return gamma.rvs(3.0, 1.0, 2.0, 10, 1)

    assert_allclose(test(), gamma.rvs(3, 1, 2, 10, 1))


def test_integrate():
    par = 2.5, 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = gamma.integrate(lo, hi, *par)
        expected = np.diff(gamma.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(gamma.integrate(-np.inf, np.inf, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return gamma.integrate(lo, hi, 2.5, 1.0, 2.0)

    expected = gamma.integrate(2.0, 5.0, 2.5, 1.0, 2.0)
    assert_allclose(test(2.0, 5.0), expected)
