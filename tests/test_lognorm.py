import numba as nb
import numpy as np
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
