import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import chi2

dfs = (0.5, 1.0, 2.0, 2.5, 4.0, 10.0, 30.0)

# includes points below and at the lower edge of the support, loc = 2
x = np.linspace(-1, 20, 22)


def test_pdf_one():
    got = chi2.pdf(3, 1, 2, 3)
    expected = sc.chi2.pdf(3, 1, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("df", dfs)
def test_pdf(df):
    got = chi2.pdf(x, df, 2, 3)
    expected = sc.chi2.pdf(x, df, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("df", dfs)
def test_logpdf(df):
    got = chi2.logpdf(x, df, 2, 3)
    expected = sc.chi2.logpdf(x, df, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("df", dfs)
def test_cdf(df):
    got = chi2.cdf(x, df, 2, 3)
    expected = sc.chi2.cdf(x, df, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("df", dfs)
def test_ppf(df):
    p = np.linspace(0, 1, 10)
    got = chi2.ppf(p, df, 2, 3)
    expected = sc.chi2.ppf(p, df, 2, 3)
    assert_allclose(got, expected)

    got = chi2.ppf(0.5, df, 2, 3)
    expected = sc.chi2.ppf(0.5, df, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1.0, 4.5))
def test_rvs(df):
    x = chi2.rvs(df, 2, 3, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: chi2.cdf(x, df, 2, 3))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [chi2.logpdf, chi2.pdf, chi2.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 3.0, 1.0, 2.0)

    x = np.linspace(1.1, 20, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 3, 1, 2))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return chi2.rvs(3.0, 1.0, 2.0, 10, 1)

    assert_allclose(test(), chi2.rvs(3, 1, 2, 10, 1))
