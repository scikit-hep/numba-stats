import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import chi2


def test_pdf_one():
    df = 1
    loc = 2
    scale = 3
    x = loc + 1
    got = chi2.pdf(x, df, loc, scale)
    expected = sc.chi2.pdf(x, df, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_pdf():
    df = 1
    loc = 2
    scale = 3
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = chi2.pdf(x, df, loc, scale)
    expected = sc.chi2.pdf(x, df, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_logpdf():
    df = 1
    loc = 2
    scale = 3
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = chi2.logpdf(x, df, loc, scale)
    expected = sc.chi2.logpdf(x, df, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_cdf():
    df = 1
    loc = 2
    scale = 3
    x = np.linspace(loc + 0.1, loc + 5, 10)
    got = chi2.cdf(x, df, loc, scale)
    expected = sc.chi2.cdf(x, df, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_ppf():
    df = 4
    loc = 3
    scale = 4

    p = np.linspace(0, 1, 10)
    got = chi2.ppf(p, df, loc, scale)
    expected = sc.chi2.ppf(p, df, loc=loc, scale=scale)
    assert_allclose(got, expected)

    got = chi2.ppf(0.5, df, loc, scale)
    expected = sc.chi2.ppf(0.5, df, loc=loc, scale=scale)
    assert_allclose(got, expected)


def test_rvs():
    df = 4
    loc = 3
    scale = 4
    x = chi2.rvs(df, loc, scale, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: chi2.cdf(x, df, loc, scale))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [chi2.logpdf, chi2.pdf, chi2.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 0.0, 1.0, 2.0)

    x = np.linspace(1.1, 5.0, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 0, 1, 2))
