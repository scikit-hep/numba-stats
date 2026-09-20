import numba as nb
import numpy as np
import pytest
import scipy.stats as sc
from numpy.testing import assert_allclose

from numba_stats import chi2, ncx2

# df, nc; includes nc = 0, which is the null hypothesis in likelihood ratio tests
pars = ((1.0, 0.5), (2.0, 1.0), (2.5, 3.0), (4.0, 2.0), (10.0, 30.0), (3.0, 0.0))

# includes points below the lower edge of the support, loc = 2
x = np.linspace(-1, 30, 30)


def test_pdf_one():
    got = ncx2.pdf(3, 4, 2, 2, 3)
    expected = sc.ncx2.pdf(3, 4, 2, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize(("df", "nc"), pars)
def test_pdf(df, nc):
    got = ncx2.pdf(x, df, nc, 2, 3)
    expected = sc.ncx2.pdf(x, df, nc, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize(("df", "nc"), pars)
def test_logpdf(df, nc):
    got = ncx2.logpdf(x, df, nc, 2, 3)
    expected = sc.ncx2.logpdf(x, df, nc, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize(("df", "nc"), pars)
def test_cdf(df, nc):
    got = ncx2.cdf(x, df, nc, 2, 3)
    expected = sc.ncx2.cdf(x, df, nc, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize(("df", "nc"), pars)
def test_ppf(df, nc):
    p = np.linspace(0, 1, 10)
    got = ncx2.ppf(p, df, nc, 2, 3)
    expected = sc.ncx2.ppf(p, df, nc, 2, 3)
    assert_allclose(got, expected)

    got = ncx2.ppf(0.5, df, nc, 2, 3)
    expected = sc.ncx2.ppf(0.5, df, nc, 2, 3)
    assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1.0, 2.0, 4.5))
def test_vs_chi2(df):
    # the distribution becomes a chi-squared distribution for nc -> 0
    assert_allclose(ncx2.pdf(x, df, 0, 2, 3), chi2.pdf(x, df, 2, 3))
    assert_allclose(ncx2.cdf(x, df, 0, 2, 3), chi2.cdf(x, df, 2, 3))


@pytest.mark.parametrize(("df", "nc"), ((1.0, 0.5), (4.0, 2.0), (3.0, 0.0)))
def test_rvs(df, nc):
    x = ncx2.rvs(df, nc, 2, 3, size=100_000, random_state=2)
    r = sc.kstest(x, lambda x: ncx2.cdf(x, df, nc, 2, 3))
    assert r.pvalue > 0.01


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [ncx2.logpdf, ncx2.pdf, ncx2.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 3.0, 2.0, 1.0, 2.0)

    x = np.linspace(1.1, 30, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 3, 2, 1, 2))


@pytest.mark.filterwarnings("error")
def test_rvs_njit():
    @nb.njit
    def test():
        return ncx2.rvs(3.0, 2.0, 1.0, 2.0, 10, 1)

    assert_allclose(test(), ncx2.rvs(3, 2, 1, 2, 10, 1))


def test_integrate():
    par = 3, 2, 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = ncx2.integrate(lo, hi, *par)
        expected = np.diff(ncx2.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(ncx2.integrate(-np.inf, np.inf, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return ncx2.integrate(lo, hi, 3.0, 2.0, 1.0, 2.0)

    expected = ncx2.integrate(2.0, 5.0, 3.0, 2.0, 1.0, 2.0)
    assert_allclose(test(2.0, 5.0), expected)
