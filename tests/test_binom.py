import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import binom

# NC and KC are all combinations of n and k from 0 to 10
N = np.arange(10)
NC = []
KC = []
for n in N:
    for k in range(n + 1):
        NC.append(n)
        KC.append(k)
NC = np.array(NC, np.float64)
KC = np.array(KC, np.float64)


@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_pmf(p):
    got = binom.pmf(KC, NC, p)
    expected = sc.binom.pmf(KC, NC, p)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_cdf(p):
    got = binom.cdf(KC, NC, p)
    expected = sc.binom.cdf(KC, NC, p)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("n", np.arange(0, 10, 2))
@pytest.mark.parametrize("p", np.linspace(0, 1, 5))
def test_rvs(n, p):
    got = binom.rvs(n, p, size=1000, random_state=1)

    @nb.njit
    def expected():
        np.random.seed(1)
        return np.random.binomial(n, p, 1000)

    np.testing.assert_equal(got, expected())


@pytest.mark.parametrize("p", (0.0, 0.3, 1.0))
def test_integrate(p):
    # probability of lo < k <= hi
    n = 20
    for lo, hi in ((0, 5), (0, 20), (3, 3), (7, 2), (10, 20)):
        got = binom.integrate(lo, hi, n, p)
        expected = sc.binom.cdf(hi, n, p) - sc.binom.cdf(lo, n, p)
        assert_allclose(got, expected, atol=1e-15)
    k = np.array([2.0, 9.0])
    got = binom.integrate(k[0], k[1], n, p)
    assert_allclose(got, np.diff(binom.cdf(k, np.full_like(k, n), p))[0], atol=1e-15)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return binom.integrate(lo, hi, 20.0, 0.3)

    assert_allclose(test(2.0, 9.0), binom.integrate(2, 9, 20.0, 0.3))
