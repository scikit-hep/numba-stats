import numba as nb
import numpy as np
import pytest
import scipy.stats as sc

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
