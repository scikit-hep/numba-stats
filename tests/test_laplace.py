import numpy as np
import scipy.stats as sc

from numba_stats import laplace


def test_pdf():
    x = np.linspace(-5, 5, 20)
    got = laplace.pdf(x, 1, 2)
    expected = sc.laplace.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-5, 5, 20) + 3
    got = laplace.cdf(x, 3, 2)
    expected = sc.laplace.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 20)
    got = laplace.ppf(p, 1, 2)
    expected = sc.laplace.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_rvs():
    args = 1, 2
    x = laplace.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: laplace.cdf(x, *args))
    assert r.pvalue > 0.01
