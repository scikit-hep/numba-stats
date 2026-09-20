import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from numba_stats import cmsshape, expon


def test_pdf_1():
    par = 1.1, 2.2, 3.3
    assert quad(lambda x: cmsshape.pdf(x, *par), -10, 10)[0] == pytest.approx(1)


def test_pdf_2():
    par = 1e3, 1.5, 0
    x = np.linspace(-3, 3, 1000)
    got = cmsshape.pdf(x, *par)
    expected = expon.pdf(x, 0, 2 / 3)
    assert_allclose(got, expected, atol=1e-3)


def test_cdf():
    par = 1.1, 2.2, 3.3
    x = np.linspace(-3, 3, 20)

    @np.vectorize
    def num_cdf(x):
        return quad(lambda x: cmsshape.pdf(x, *par), -10, x)[0]

    expected = num_cdf(x)
    got = cmsshape.cdf(x, *par)
    assert_allclose(got, expected, atol=1e-10)


def test_integrate():
    par = 1, 2, 0.5
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = cmsshape.integrate(lo, hi, *par)
        expected = np.diff(cmsshape.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return cmsshape.integrate(lo, hi, 1.0, 2.0, 0.5)

    expected = cmsshape.integrate(-1.0, 1.5, 1.0, 2.0, 0.5)
    assert_allclose(test(-1.0, 1.5), expected)
