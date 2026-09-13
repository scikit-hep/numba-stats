import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import laplace


def test_pdf():
    x = np.linspace(-5, 5, 20)
    got = laplace.pdf(x, 1, 2)
    expected = sc.laplace.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-5, 5, 20) + 3
    got = laplace.cdf(x, 3, 2)
    expected = sc.laplace.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 20)
    got = laplace.ppf(p, 1, 2)
    expected = sc.laplace.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_rvs():
    args = 1, 2
    x = laplace.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: laplace.cdf(x, *args))
    assert r.pvalue > 0.01


def test_integrate():
    par = 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = laplace.integrate(lo, hi, *par)
        expected = np.diff(laplace.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(laplace.integrate(-np.inf, np.inf, *par), 1)


def test_integrate_tail():
    got = laplace.integrate(30, 31, 0, 1)
    assert_allclose(got, sc.laplace.sf(30) - sc.laplace.sf(31), rtol=1e-10)
    got = laplace.integrate(-31, -30, 0, 1)
    assert_allclose(got, sc.laplace.cdf(-30) - sc.laplace.cdf(-31), rtol=1e-10)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return laplace.integrate(lo, hi, 1.0, 2.0)

    expected = laplace.integrate(-1.0, 1.5, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
