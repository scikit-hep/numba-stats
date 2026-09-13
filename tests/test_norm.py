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


def test_integrate():
    par = 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = norm.integrate(lo, hi, *par)
        assert isinstance(got, float)
        expected = np.diff(norm.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(norm.integrate(-np.inf, np.inf, *par), 1)


def test_integrate_tail():
    # cdf(hi) - cdf(lo) loses relative precision in the tails, integrate does not
    got = norm.integrate(7, 8, 0, 1)
    assert_allclose(got, sc.norm.sf(7) - sc.norm.sf(8), rtol=1e-10)
    got = norm.integrate(-8, -7, 0, 1)
    assert_allclose(got, sc.norm.cdf(-7) - sc.norm.cdf(-8), rtol=1e-10)


def test_integrate_doc():
    assert "lo : float" in norm.integrate.__doc__
    assert "Integral of the density" in norm.integrate.__doc__


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return norm.integrate(lo, hi, 1.0, 2.0)

    expected = norm.integrate(-1.0, 1.5, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
    # integer limits are accepted in compiled code
    assert_allclose(test(-1, 1), norm.integrate(-1, 1, 1.0, 2.0))


@pytest.mark.filterwarnings("error")
def test_integrate_njit_float32():
    @nb.njit
    def test(lo, hi):
        return norm.integrate(lo, hi, np.float32(1), np.float32(2))

    got = test(np.float32(-1), np.float32(1.5))
    assert isinstance(got, float)
    assert_allclose(got, norm.integrate(-1, 1.5, 1, 2), rtol=1e-6)
