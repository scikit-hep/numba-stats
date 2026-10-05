import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats as sc
from scipy.integrate import quad

from numba_stats import crystalball as cb


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.1, 2, 3))
def test_powerlaw_integral(beta, m):
    expected = quad(lambda z: np.exp(cb._log_powerlaw(z, beta, m)), -np.inf, -beta)[0]
    got = cb._powerlaw_integral(-beta, beta, m)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("z", (0, 1, 10, np.inf))
def test_normal_integral(beta, z):
    expected = quad(lambda z: np.exp(-0.5 * z**2), -beta, z)[0]
    got = cb._normal_integral(-beta, z)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_logpdf(beta, m, loc):
    scale = 1.5
    x = np.linspace(-10, 5, 10)
    got = cb.logpdf(x, beta, m, loc, scale)
    expected = sc.crystalball.logpdf(x, beta, m, loc, scale)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_pdf(beta, m, loc):
    scale = 1.5
    x = np.linspace(-10, 5, 10)
    got = cb.pdf(x, beta, m, loc, scale)
    expected = sc.crystalball.pdf(x, beta, m, loc, scale)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_cdf(beta, m, loc):
    scale = 1.5
    x = np.linspace(-10, 5, 10)
    got = cb.cdf(x, beta, m, loc, scale)
    expected = sc.crystalball.cdf(x, beta, m, loc, scale)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_ppf(beta, m, loc):
    scale = 1.5
    p = np.linspace(0, 1, 10)
    p = np.append(p, [0.5, 1.01])
    got = cb.ppf(p, beta, m, loc, scale)
    with np.errstate(over="ignore"):
        expected = sc.crystalball.ppf(p, beta, m, loc, scale)
    assert_allclose(got, expected)


def test_integrate():
    par = 1, 3, 1, 2
    for lo, hi in ((-3.0, 4.0), (0.5, 0.7), (4.0, -3.0), (-100.0, 100.0), (2.5, 6.0)):
        got = cb.integrate(lo, hi, *par)
        expected = np.diff(cb.cdf([lo, hi], *par))[0]
        assert_allclose(got, expected, rtol=1e-9, atol=1e-15)
    assert_allclose(cb.integrate(-np.inf, np.inf, *par), 1)


@pytest.mark.filterwarnings("error")
def test_integrate_njit():
    @nb.njit
    def test(lo, hi):
        return cb.integrate(lo, hi, 1.0, 3.0, 1.0, 2.0)

    expected = cb.integrate(-1.0, 1.5, 1.0, 3.0, 1.0, 2.0)
    assert_allclose(test(-1.0, 1.5), expected)
