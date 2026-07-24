import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import gamma


def test_pdf_one():
    alpha = 1
    loc = 2
    scale = 3
    x = loc + 1
    got = gamma.pdf(x, alpha, loc, scale)
    expected = sc.gamma.pdf(x, alpha, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_pdf():
    alpha = 1
    loc = 2
    scale = 3
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = gamma.pdf(x, alpha, loc, scale)
    expected = sc.gamma.pdf(x, alpha, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_logpdf():
    alpha = 1
    loc = 2
    scale = 3
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = gamma.logpdf(x, alpha, loc, scale)
    expected = sc.gamma.logpdf(x, alpha, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_cdf():
    alpha = 1
    loc = 2
    scale = 3
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = gamma.cdf(x, alpha, loc, scale)
    expected = sc.gamma.cdf(x, alpha, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_ppf():
    alpha = 4
    loc = 3
    scale = 4

    p = np.linspace(0, 1, 10)
    got = gamma.ppf(p, alpha, loc, scale)
    expected = sc.gamma.ppf(p, alpha, loc=loc, scale=scale)
    assert_allclose(got, expected)

    got = gamma.ppf(0.5, alpha, loc, scale)
    expected = sc.gamma.ppf(0.5, alpha, loc=loc, scale=scale)
    assert_allclose(got, expected)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [gamma.logpdf, gamma.pdf, gamma.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 0.0, 1.0, 2.0)

    x = np.linspace(1.1, 5.0, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 0, 1, 2))
