import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats as sc

from numba_stats import argus


@pytest.mark.parametrize("chi", (0.1, 0.5, 1.0, 2.0, 3.0))
def test_logpdf(chi):
    c = 1
    p = 0.5
    x = np.linspace(-1, 2, 30)
    got = argus.logpdf(x, chi, c, p)
    expected = sc.argus.logpdf(x, chi)
    assert_allclose(got, expected)


@pytest.mark.parametrize("chi", (0.1, 0.5, 1.0, 2.0, 3.0))
def test_pdf(chi):
    c = 1
    p = 0.5
    x = np.linspace(-1, 2, 30)
    got = argus.pdf(x, chi, c, p)
    expected = sc.argus.pdf(x, chi)
    assert_allclose(got, expected)


@pytest.mark.parametrize("chi", (0.1, 0.5, 1.0, 2.0, 3.0))
def test_cdf(chi):
    c = 1
    p = 0.5
    x = np.linspace(-1, 2, 30)
    got = argus.cdf(x, chi, c, p)
    expected = sc.argus.cdf(x, chi)
    assert_allclose(got, expected, atol=2e-16)


def test_integrate():
    par = 1, 2, 0.5
    for lo, hi in ((0.1, 1.5), (0.5, 0.7), (1.5, 0.1), (-1.0, 3.0)):
        got = argus.integrate(lo, hi, *par)
        expected = np.diff(argus.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(argus.integrate(0, 2, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return argus.integrate(lo, hi, 1.0, 2.0, 0.5)

    expected = argus.integrate(0.0, 1.0, 1.0, 2.0, 0.5)
    assert_allclose(test(0.0, 1.0), expected)
