import csv
from pathlib import Path

import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import voigt_profile

from numba_stats import norm, voigt

# Values for loc = 0 and scale = 1, computed with mpmath by
# make_voigt_cdf_reference.py from two independent methods that agree to 1e-25.
CDF_REFERENCE = Path(__file__).parent / "voigt_cdf_reference.csv"

# The cdf switches from quadrature to the asymptotic expansion at |z| = 6,
# with z = (x - loc + i gamma) / (scale sqrt(2)).
CDF_SWITCH = 6.0

GAMMAS = (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def test_pdf():
    gamma = 2
    sigma = 1
    mu = -1
    x = np.linspace(-5, 5, 10)
    got = voigt.pdf(x, gamma, mu, sigma)
    # note that sigma comes before gamma in scipy
    expected = voigt_profile(x - mu, sigma, gamma)
    np.testing.assert_allclose(got, expected)


@pytest.fixture(scope="module")
def cdf_reference():
    with CDF_REFERENCE.open() as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return tuple(
        np.array([float(row[key]) for row in rows]) for key in ("gamma", "x", "cdf")
    )


def _outside_core(x, gamma):
    return np.abs(x + 1j * gamma) / np.sqrt(2) >= CDF_SWITCH


def test_cdf_against_reference(cdf_reference):
    gammas, xs, expected = cdf_reference
    for gamma in np.unique(gammas):
        m = gammas == gamma
        got = voigt.cdf(xs[m], gamma, 0, 1)
        assert_allclose(got, expected[m], rtol=1e-12, atol=1e-14, err_msg=f"{gamma=}")


def test_cdf_left_tail_relative_precision(cdf_reference):
    # Small left-tail values outside the core keep their relative precision. The
    # bound is set by the expansion leaving out the Gaussian tail of the profile,
    # which is 1e-13 of the cdf just outside the switch at gamma = 0.001.
    gammas, xs, expected = cdf_reference
    outside = (xs < 0) & _outside_core(xs, gammas)
    for gamma in np.unique(gammas[outside]):
        m = outside & (gammas == gamma)
        got = voigt.cdf(xs[m], gamma, 0, 1)
        assert_allclose(got, expected[m], rtol=1e-12, atol=0, err_msg=f"{gamma=}")


def test_cdf_loc_scale(cdf_reference):
    # the cdf depends only on (x - loc) / scale and gamma / scale
    gammas, xs, expected = cdf_reference
    loc, scale = 3.0, 2.5
    for gamma in np.unique(gammas):
        m = gammas == gamma
        got = voigt.cdf(loc + scale * xs[m], gamma * scale, loc, scale)
        assert_allclose(got, expected[m], rtol=1e-12, atol=1e-14, err_msg=f"{gamma=}")


@pytest.mark.parametrize("gamma", GAMMAS)
def test_cdf_symmetric(gamma):
    assert_allclose(voigt.cdf(0, gamma, 0, 1), 0.5, rtol=0, atol=1e-15)
    d = np.geomspace(1e-3, 1e4, 200)
    got = voigt.cdf(d, gamma, 0, 1) + voigt.cdf(-d, gamma, 0, 1)
    assert_allclose(got, 1, rtol=0, atol=1e-14)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_cdf_monotonic_with_cauchy_tails(gamma):
    d = np.geomspace(1e-3, 1e8, 500)
    x = np.concatenate([-d[::-1], [0], d])
    got = voigt.cdf(x, gamma, 0, 1)
    assert np.all(np.diff(got) >= 0)
    # far out, the cdf approaches the Cauchy tail gamma / (pi |x|); on the right,
    # the cdf is close to 1, so 1 - cdf has only absolute precision
    tail = gamma / (np.pi * 1e8)
    assert_allclose(got[0], tail, rtol=1e-6)
    assert_allclose(1 - got[-1], tail, rtol=0, atol=1e-15)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_cdf_derivative_is_pdf(gamma):
    x = np.linspace(-30, 30, 121)
    h = 1e-4
    got = (voigt.cdf(x + h, gamma, 0, 1) - voigt.cdf(x - h, gamma, 0, 1)) / (2 * h)
    expected = voigt.pdf(x, gamma, 0, 1)
    assert_allclose(got, expected, rtol=1e-6, atol=1e-11)


@pytest.mark.parametrize("gamma", (0.001, 0.3, 1.0, 3.0, 8.0))
def test_cdf_continuous_at_switch(gamma):
    # just inside and just outside the switch, the branches must agree to within
    # the change of the cdf
    y = gamma / np.sqrt(2)
    edge = np.sqrt(2 * (CDF_SWITCH**2 - y**2))
    eps = 1e-9
    step = voigt.pdf(edge, gamma, 0, 1) * 2 * eps
    for sign in (1, -1):
        inside, outside = voigt.cdf(sign * (edge + np.array([-eps, eps])), gamma, 0, 1)
        assert_allclose(abs(outside - inside), step, rtol=0, atol=1e-14)


def test_cdf_gamma_zero_is_norm():
    x = np.linspace(-15, 15, 301)
    got = voigt.cdf(x, 0, 1, 2)
    expected = norm.cdf(x, 1, 2)
    assert_allclose(got, expected, rtol=0, atol=1e-14)


def test_cdf_float32():
    x = np.linspace(-20, 20, 41)
    got = voigt.cdf(x.astype(np.float32), np.float32(1), np.float32(0), np.float32(2))
    assert got.dtype == np.float32
    assert_allclose(got, voigt.cdf(x, 1, 0, 2), rtol=1e-6)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("fn", [voigt.pdf, voigt.cdf])
@pytest.mark.parametrize("parallel", [False, True])
def test_njit(fn, parallel):
    @nb.njit(parallel=parallel, fastmath=True)
    def test(x):
        return fn(x, 1.0, 0.5, 2.0)

    x = np.linspace(-30, 30, 1000)
    y = test(x)

    assert_allclose(y, fn(x, 1, 0.5, 2))
