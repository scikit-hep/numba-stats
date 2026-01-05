import numpy as np
import pytest

from numba_stats import cpoisson, poisson


@pytest.mark.parametrize("mu", (0.1, 0.5, 1.0, 2.0))
def test_cdf(mu):
    k = np.arange(10)
    got = cpoisson.cdf(k, mu)
    expected = poisson.cdf(k, mu)
    np.testing.assert_allclose(got, expected)
