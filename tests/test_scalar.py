import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose

from numba_stats import crystalball, norm, poisson, uniform


# see https://github.com/scikit-hep/numba-stats/issues/56
@pytest.mark.filterwarnings("error")
def test_scalar_njit():
    @nb.njit
    def prior():
        p = 1.0
        for _ in range(3):
            p *= uniform.pdf(0.5, 0.0, 1.0)
        return p

    assert prior() == 1.0


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("fn", "x"),
    [
        (lambda x: norm.pdf(x, 0.0, 1.0), 0.5),
        (lambda x: norm.logpdf(x, 0.0, 1.0), 0.5),
        (lambda x: norm.cdf(x, 0.0, 1.0), 0.5),
        (lambda x: norm.ppf(x, 0.0, 1.0), 0.3),
        (lambda x: uniform.cdf(x, 0.0, 1.0), 0.5),
        (lambda x: uniform.ppf(x, 0.0, 1.0), 0.3),
        (lambda x: crystalball.pdf(x, 1.0, 3.0, 0.0, 1.0), -2.0),
        (lambda x: poisson.pmf(x, 3.0), 2.0),
    ],
)
def test_scalar_njit_vs_array(fn, x):
    f = nb.njit(fn)
    expected = fn(np.array([x]))[0]
    assert_allclose(f(x), expected)
    assert isinstance(f(x), float)


@pytest.mark.filterwarnings("error")
def test_scalar_int_njit():
    @nb.njit
    def f(k):
        return poisson.pmf(k, 3.0)

    assert_allclose(f(2), poisson.pmf(2, 3.0))


@pytest.mark.filterwarnings("error")
def test_scalar_float32_njit():
    @nb.njit
    def f(x):
        return norm.pdf(x, np.float32(0), np.float32(1))

    x = np.float32(0.5)
    got = f(x)
    assert isinstance(got, float)
    assert_allclose(got, norm.pdf(np.array([x]), np.float32(0), np.float32(1))[0])
