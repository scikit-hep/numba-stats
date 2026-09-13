import numba as nb
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.interpolate import BPoly

from numba_stats import bernstein


def scipy_density(x, beta, xmin, xmax):
    return BPoly(np.array(beta)[:, np.newaxis], [xmin, xmax])(x)


@pytest.mark.parametrize(
    "beta", [[1.0], [1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 2.0]]
)
def test_density(beta):
    x = np.linspace(1, 3, 10000)
    got = bernstein.density(x, beta, x[0], x[-1])
    expected = scipy_density(x, beta, x[0], x[-1])
    assert_allclose(got, expected)

    got = bernstein.density(0.5, beta, 0, 1)
    expected = bernstein.density([0.5], beta, 0, 1)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", [[1], [1, 1], [1, 1, 1]])
def test_integral(beta):
    xrange = 1.5, 3.4
    got = bernstein.integral(xrange[1], beta, *xrange)
    expected = np.diff(xrange)
    assert_allclose(got, expected)


@pytest.mark.parametrize("beta", [[1], [1, 1], [1, 1, 1], [1, 2, 3], [1, 3, 2]])
def test_integral_2(beta):
    x = np.linspace(1, 2.5)

    got = bernstein.integral(x, beta, x[0], x[-1])
    expected = [
        quad(lambda y: bernstein.density(y, beta, x[0], x[-1]), x[0], xi)[0] for xi in x
    ]
    assert_allclose(got, expected)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("parallel", (False, True))
@pytest.mark.parametrize("fn", [bernstein.density, bernstein.integral])
def test_numba(fn, parallel):
    x = np.linspace(0.5, 0.6, 10000)
    beta = np.array([1.0, 2.0, 3.0])
    xmin = 1.0
    xmax = 2.5

    @nb.njit(parallel=parallel, fastmath=True)
    def f():
        return fn(x, beta, xmin, xmax)

    assert_allclose(f(), fn(x, beta, xmin, xmax))


def test_deprecation():
    with pytest.warns(FutureWarning):
        got = bernstein.scaled_pdf(1, [1, 2], 0, 1)
    assert_allclose(got, bernstein.density(1, [1, 2], 0, 1))

    with pytest.warns(FutureWarning):
        got = bernstein.scaled_cdf(1, [1, 2], 0, 1)
    assert_allclose(got, bernstein.integral(1, [1, 2], 0, 1))
