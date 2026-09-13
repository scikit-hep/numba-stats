import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import lognorm


def test_pdf():
    x = np.linspace(0, 5, 10)
    got = lognorm.pdf(x, 1.5, 0.1, 1.2)
    expected = sc.lognorm.pdf(x, 1.5, 0.1, 1.2)
    assert_allclose(got, expected)


def test_logpdf():
    x = np.linspace(0, 5, 10)
    got = lognorm.logpdf(x, 1.5, 0.1, 1.2)
    expected = sc.lognorm.logpdf(x, 1.5, 0.1, 1.2)
    assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(0, 5, 10)
    got = lognorm.cdf(x, 1.5, 0.1, 1.2)
    expected = sc.lognorm.cdf(x, 1.5, 0.1, 1.2)
    assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 10)
    got = lognorm.ppf(p, 1.5, 0.1, 1.2)
    expected = sc.lognorm.ppf(p, 1.5, 0.1, 1.2)
    assert_allclose(got, expected)


def test_rvs():
    args = 1.5, 0.1, 1.2
    x = lognorm.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: lognorm.cdf(x, *args))
    assert r.pvalue > 0.01


def test_njit():
    @nb.njit
    def test(x):
        a = lognorm.logpdf(x, 1.0, 0.0, 1.0)
        b = lognorm.pdf(x, 1.0, 0.0, 1.0)
        c = lognorm.cdf(x, 1.0, 0.0, 1.0)
        d = lognorm.ppf(c, 1.0, 0.0, 1.0)
        return a, b, c, d

    x = np.linspace(0, 3, 10)
    a, b, c, d = test(x)

    assert_allclose(a, lognorm.logpdf(x, 1.0, 0.0, 1.0))
    assert_allclose(b, lognorm.pdf(x, 1.0, 0.0, 1.0))
    assert_allclose(c, lognorm.cdf(x, 1.0, 0.0, 1.0))
    assert_allclose(d, x)


def test_integrate():
    par = 0.5, 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = lognorm.integrate(lo, hi, *par)
        expected = np.diff(lognorm.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(lognorm.integrate(-np.inf, np.inf, *par), 1)


def test_integrate_tail():
    got = lognorm.integrate(1e5, 2e5, 0.5, 0, 1)
    assert_allclose(got, sc.lognorm.sf(1e5, 0.5) - sc.lognorm.sf(2e5, 0.5), rtol=1e-10)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return lognorm.integrate(lo, hi, 0.5, 1.0, 2.0)

    expected = lognorm.integrate(-1.0, 1.5, 0.5, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
