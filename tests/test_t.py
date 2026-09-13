import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import t


@pytest.mark.parametrize("df", (1, 1.5, 2, 3, 4, 5.5, 10))
def test_pdf(df):
    x = np.linspace(-5, 5, 10)
    got = t.pdf(x, df, 2, 3)
    expected = sc.t.pdf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_cdf(df):
    x = np.linspace(-5, 5, 10)
    got = t.cdf(x, df, 2, 3)
    expected = sc.t.cdf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_ppf(df):
    x = np.linspace(0, 1, 10)
    got = t.ppf(x, df, 2, 3)
    expected = sc.t.ppf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_rvs(df):
    args = df, 2, 3
    x = t.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: t.cdf(x, *args))
    assert r.pvalue > 0.01


def test_integrate():
    par = 3, 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = t.integrate(lo, hi, *par)
        expected = np.diff(t.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(t.integrate(-np.inf, np.inf, *par), 1)


def test_integrate_tail():
    got = t.integrate(50, 60, 3, 0, 1)
    assert_allclose(got, sc.t.sf(50, 3) - sc.t.sf(60, 3), rtol=1e-10)
    got = t.integrate(-60, -50, 3, 0, 1)
    assert_allclose(got, sc.t.cdf(-50, 3) - sc.t.cdf(-60, 3), rtol=1e-10)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return t.integrate(lo, hi, 3.0, 1.0, 2.0)

    expected = t.integrate(-1.0, 1.5, 3.0, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
