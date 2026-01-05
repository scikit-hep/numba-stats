import numpy as np
import pytest
import scipy.stats as sc

from numba_stats import t


@pytest.mark.parametrize("df", (1, 1.5, 2, 3, 4, 5.5, 10))
def test_pdf(df):
    x = np.linspace(-5, 5, 10)
    got = t.pdf(x, df, 2, 3)
    expected = sc.t.pdf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_cdf(df):
    x = np.linspace(-5, 5, 10)
    got = t.cdf(x, df, 2, 3)
    expected = sc.t.cdf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_ppf(df):
    x = np.linspace(0, 1, 10)
    got = t.ppf(x, df, 2, 3)
    expected = sc.t.ppf(x, df, 2, 3)  # supports real-valued df
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("df", (1, 1.5, 3, 5.5, 10))
def test_rvs(df):
    args = df, 2, 3
    x = t.rvs(*args, size=100_000, random_state=1)
    r = sc.kstest(x, lambda x: t.cdf(x, *args))
    assert r.pvalue > 0.01
