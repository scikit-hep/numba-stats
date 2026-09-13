import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import poisson


@pytest.mark.parametrize("mu", np.linspace(0, 3, 5))
def test_pmf(mu):
    k = np.arange(10)
    got = poisson.pmf(k, mu)
    expected = sc.poisson.pmf(k, mu)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("mu", np.linspace(0, 3, 5))
def test_cdf(mu):
    k = np.arange(10)
    got = poisson.cdf(k, mu)
    expected = sc.poisson.cdf(k, mu)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("mu", np.linspace(0, 3, 5))
def test_rvs(mu):
    got = poisson.rvs(mu, size=1000, random_state=1)

    @nb.njit
    def expected():
        np.random.seed(1)
        return np.random.poisson(mu, 1000)

    np.testing.assert_equal(got, expected())


@pytest.mark.parametrize("mu", (0.5, 3.0, 10.0))
def test_integrate(mu):
    # probability of lo < k <= hi
    for lo, hi in ((-1, 3), (0, 5), (2, 2), (5, 0), (3, 40), (-1, 100)):
        got = poisson.integrate(lo, hi, mu)
        expected = sc.poisson.cdf(hi, mu) - sc.poisson.cdf(lo, mu)
        assert_allclose(got, expected, atol=1e-15)
        assert_allclose(got, np.diff(poisson.cdf([lo, hi], mu))[0], atol=1e-15)


def test_integrate_tail():
    got = poisson.integrate(40, 50, 3)
    assert_allclose(got, sc.poisson.sf(40, 3) - sc.poisson.sf(50, 3), rtol=1e-10)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return poisson.integrate(lo, hi, 3.0)

    assert_allclose(test(1.0, 4.0), poisson.integrate(1, 4, 3.0))
