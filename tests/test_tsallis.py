import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import tsallis


def test_pdf():
    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), 0, np.inf)
                assert abs(1 - v) < err


def test_cdf():
    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                for ptrange in ((0, 500), (500, 1000), (1000, 2000)):
                    v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), *ptrange)
                    v2 = np.diff(tsallis.cdf(ptrange, m, t, n))
                    assert abs(v2 - v) < err


def test_integrate():
    par = 1, 2, 5
    for lo, hi in ((0.1, 5.0), (0.5, 0.7), (5.0, 0.1), (-1.0, 100.0), (2.5, 6.0)):
        got = tsallis.integrate(lo, hi, *par)
        expected = np.diff(tsallis.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return tsallis.integrate(lo, hi, 1.0, 2.0, 5.0)

    expected = tsallis.integrate(1.0, 2.0, 1.0, 2.0, 5.0)
    assert_allclose(test(1.0, 2.0), expected)
