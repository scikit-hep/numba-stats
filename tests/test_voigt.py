import numpy as np
from scipy.special import voigt_profile

from numba_stats import voigt


def test_pdf():
    gamma = 2
    sigma = 1
    mu = -1
    x = np.linspace(-5, 5, 10)
    got = voigt.pdf(x, gamma, mu, sigma)
    # note that sigma comes before gamma in scipy
    expected = voigt_profile(x - mu, sigma, gamma)
    np.testing.assert_allclose(got, expected)
