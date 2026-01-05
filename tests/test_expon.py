import numpy as np
import scipy.stats as sc

from numba_stats import expon


def test_pdf():
    x = np.linspace(-5, 5, 20)
    got = expon.pdf(x, 1, 2)
    expected = sc.expon.pdf(x, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_cdf():
    x = np.linspace(-5, 5, 20) + 3
    got = expon.cdf(x, 3, 2)
    expected = sc.expon.cdf(x, 3, 2)
    np.testing.assert_allclose(got, expected)


def test_ppf():
    p = np.linspace(0, 1, 20)
    got = expon.ppf(p, 1, 2)
    expected = sc.expon.ppf(p, 1, 2)
    np.testing.assert_allclose(got, expected)


def test_rvs():
    args = 1, 2
    x = expon.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: expon.cdf(x, *args))
    assert r.pvalue > 0.01
