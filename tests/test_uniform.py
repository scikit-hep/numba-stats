import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import uniform


def test_pdf():
    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.pdf(x, -1, 2)
    expected = sc.uniform.pdf(x, -1, 2)
    assert_allclose(got, expected)

    got = uniform.pdf(1, -1, 3)
    expected = sc.uniform.pdf(1, -1, 3)
    assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-1.1, 1.1, 10)
    got = uniform.cdf(x, -1, 2)
    expected = sc.uniform.cdf(x, -1, 2)
    assert_allclose(got, expected)

    got = uniform.cdf(1, -1, 3)
    expected = sc.uniform.cdf(1, -1, 3)
    assert_allclose(got, expected)


def test_ppf():
    x = np.linspace(0, 1, 10)
    got = uniform.ppf(x, -1, 2)
    expected = sc.uniform.ppf(x, -1, 2)
    assert_allclose(got, expected)

    got = uniform.ppf(0.5, -1, 3)
    expected = sc.uniform.ppf(0.5, -1, 3)
    assert_allclose(got, expected)


def test_rvs():
    args = -1, 2
    x = uniform.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: uniform.cdf(x, *args))
    assert r.pvalue > 0.01


def test_integrate():
    par = 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = uniform.integrate(lo, hi, *par)
        expected = np.diff(uniform.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(uniform.integrate(1, 3, *par), 1)
    assert_allclose(uniform.integrate(-10, 10, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return uniform.integrate(lo, hi, 1.0, 2.0)

    expected = uniform.integrate(-1.0, 1.5, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
