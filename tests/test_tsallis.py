import numpy as np
from scipy.integrate import quad

from numba_stats import tsallis


def test_pdf():
    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), 0, np.inf)
                assert abs(1 - v) < err


def test_cdf():
    for m in (100, 1000):
        for t in (100, 1000):
            for n in (3, 5, 8):
                for ptrange in ((0, 500), (500, 1000), (1000, 2000)):
                    v, err = quad(lambda pt: tsallis.pdf(pt, m, t, n), *ptrange)
                    v2 = np.diff(tsallis.cdf(ptrange, m, t, n))
                    assert abs(v2 - v) < err
