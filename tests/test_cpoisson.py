import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gammainc

from numba_stats import cpoisson, poisson


@pytest.mark.parametrize("mu", (0.1, 0.5, 1.0, 2.0))
def test_cdf(mu):
    k = np.arange(10)
    got = cpoisson.cdf(k, mu)
    expected = poisson.cdf(k, mu)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("mu", (0.5, 3.0, 10.0))
def test_integrate(mu):
    for lo, hi in ((-0.5, 3.5), (0.2, 5.7), (2.5, 2.5), (5.5, 0.5), (3.0, 40.0)):
        got = cpoisson.integrate(lo, hi, mu)
        expected = np.diff(cpoisson.cdf([lo, hi], mu))[0]
        assert_allclose(got, expected, atol=1e-15)


def test_integrate_tail():
    # cdf(hi) - cdf(lo) loses relative precision here, the survival function of
    # the continuous Poisson distribution is the lower incomplete gamma function
    got = cpoisson.integrate(40.5, 50.5, 3)
    expected = gammainc(41.5, 3) - gammainc(51.5, 3)
    assert_allclose(got, expected, rtol=1e-10)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return cpoisson.integrate(lo, hi, 3.0)

    assert_allclose(test(1.5, 4.5), cpoisson.integrate(1.5, 4.5, 3.0))
