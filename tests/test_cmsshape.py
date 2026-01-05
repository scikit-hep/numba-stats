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
