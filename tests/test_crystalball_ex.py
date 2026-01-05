import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats as sc
from scipy.integrate import quad

from numba_stats import crystalball_ex as cb

# We verify the pdf of the generalized Crystal Ball
# piecewise using the normal scipy Crystal Ball.
# 1) Up to the loc, the lower side looks like a Crystal Ball, up to a normalization.
# 2) At loc, the density is continuous.
# 3) Above loc, the upper side looks like a mirrored Crystal Ball,
# up to a normalization.


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_pdf_left(beta, m, loc):
    scale = 1.5
    x = np.linspace(-10, loc, 10)
    got = cb.pdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc)
    expected = sc.crystalball.pdf(x, beta, m, loc, scale)
    expected *= got[-1] / expected[-1]
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", [-0.1, 0.0, 0.1])
def test_pdf_right(beta, m, loc):
    scale = 1.5
    x = np.linspace(loc, 10, 10)
    got = cb.pdf(x, 2 * beta, 2 * m, 2 * scale, beta, m, scale, loc)
    expected = sc.crystalball.pdf(-x + 2 * loc, beta, m, loc, scale)
    expected *= got[0] / expected[0]
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_pdf_continues_at_loc(beta, m, loc):
    scale = 1.5
    eps = 1e-4
    x = np.array([loc - eps, loc + eps])
    got = cb.pdf(x, 2 * beta, 2 * m, 2 * scale, beta, m, scale, loc)
    expected = cb.pdf(loc, 2 * beta, 2 * m, 2 * scale, beta, m, scale, loc)
    assert got[0] < expected
    assert got[1] < expected
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_pdf_integral(beta, m, loc):
    scale = 1.5
    got = quad(
        lambda x: cb.pdf(x, 2 * beta, 2 * m, 2 * scale, beta, m, scale, loc),
        -np.inf,
        np.inf,
    )[0]
    assert_allclose(got, 1)


@pytest.mark.parametrize("beta", (5, 1, 0.1))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_logpdf(beta, m, loc):
    scale = 1.5
    x = np.linspace(-10, 10, 10)
    got = cb.logpdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc)
    # we can use pdf as reference, because pdf is verified against scipy
    expected = np.log(cb.pdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc))
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_cdf(beta, m, loc):
    scale = 1.5
    x = np.linspace(-10, 10, 10)
    got = cb.cdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc)
    expected = [
        quad(
            lambda x: cb.pdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc),
            -np.inf,
            xi,
        )[0]
        for xi in x
    ]
    assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("beta", (0.1, 2, 3))
@pytest.mark.parametrize("m", (1.001, 2, 3))
@pytest.mark.parametrize("loc", (-0.1, 0, 0.1))
def test_ppf(beta, m, loc):
    # We verify the ppf by checking that it is the inverse of the cdf.
    scale = 1.5
    x = np.linspace(-10, 10, 100)
    p = cb.cdf(x, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc)
    x2 = cb.ppf(p, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc)
    assert_allclose(x, x2)
    assert cb.ppf(0, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc) == -np.inf
    assert cb.ppf(1, beta, m, scale, 2 * beta, 2 * m, 2 * scale, loc) == np.inf
