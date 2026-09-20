import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import expon


def test_pdf():
    x = np.linspace(-5, 5, 20)
    got = expon.pdf(x, 1, 2)
    expected = sc.expon.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-5, 5, 20) + 3
    got = expon.cdf(x, 3, 2)
    expected = sc.expon.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 20)
    got = expon.ppf(p, 1, 2)
    expected = sc.expon.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_rvs():
    args = 1, 2
    x = expon.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: expon.cdf(x, *args))
    assert r.pvalue > 0.01


def test_integrate():
    par = 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = expon.integrate(lo, hi, *par)
        expected = np.diff(expon.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(expon.integrate(-np.inf, np.inf, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return expon.integrate(lo, hi, 1.0, 2.0)

    expected = expon.integrate(-1.0, 1.5, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
