import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import ncx2


def test_pdf_one():
    df = 4
    nc = 2
    loc = 3
    scale = 4
    x = loc + 1
    got = ncx2.pdf(x, df, nc, loc, scale)
    expected = sc.ncx2.pdf(x, df, nc, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_pdf():
    df = 4
    nc = 2
    loc = 3
    scale = 4
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = ncx2.pdf(x, df, nc, loc, scale)
    expected = sc.ncx2.pdf(x, df, nc, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_logpdf():
    df = 4
    nc = 2
    loc = 3
    scale = 4
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = ncx2.logpdf(x, df, nc, loc, scale)
    expected = sc.ncx2.logpdf(x, df, nc, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_cdf():
    df = 4
    nc = 2
    loc = 3
    scale = 4
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = ncx2.cdf(x, df, nc, loc, scale)
    expected = sc.ncx2.cdf(x, df, nc, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_ppf():
    df = 4
    nc = 2
    loc = 3
    scale = 4

    p = np.linspace(0, 1, 10)
    got = ncx2.ppf(p, df, nc, loc, scale)
    expected = sc.ncx2.ppf(p, df, nc, loc=loc, scale=scale)
    assert_allclose(got, expected)

    got = ncx2.ppf(0.5, df, nc, loc, scale)
    expected = sc.ncx2.ppf(0.5, df, nc, loc=loc, scale=scale)
    assert_allclose(got, expected)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [ncx2.logpdf, ncx2.pdf, ncx2.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 0.0, 1.0, 2.0, 3.0)

    x = np.linspace(1.1, 5.0, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 0, 1, 2, 3))
