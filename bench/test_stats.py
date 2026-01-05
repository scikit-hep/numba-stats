import importlib

import numba as nb
import numpy as np
import pytest
import scipy.stats

N = (10, 100, 1000, 10_000, 100_000)

KIND = (
    "norm.pdf",
    "norm.cdf",
    "norm.ppf",
    "truncnorm.pdf",
    "truncnorm.cdf",
    "truncnorm.ppf",
    "expon.pdf",
    "expon.cdf",
    "expon.ppf",
    "truncexpon.pdf",
    "truncexpon.cdf",
    "truncexpon.ppf",
    "t.pdf",
    "t.cdf",
    "t.ppf",
    "uniform.pdf",
    "uniform.cdf",
    "uniform.ppf",
)

ARGS = {
    "norm": (0.1, 1.2),
    "truncnorm": (-1.0, 1.0, 0.0, 1.0),
    "expon": (0.0, 1.4),
    "truncexpon": (0.0, 1.0, 0.0, 1.0),
    "t": (5.0, 0.0, 1.0),
    "uniform": (0.0, 1.0),
}


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("lib", ("scipy", "ours", "ours:parallel,fastmath"))
@pytest.mark.parametrize("kind", KIND)
def test_speed(benchmark, lib, kind, n):
    dist, meth = kind.split(".")

    if kind == "t.ppf" and lib == "ours:parallel,fastmath":
        pytest.skip("broken in parallel mode")

    if meth == "ppf":
        x = np.linspace(0, 1, n)
    else:
        x = np.linspace(-5 if dist != "expon" else 0, 5, n)

    rng = np.random.default_rng(1)
    rng.shuffle(x)

    args = ARGS[dist]
    if dist == "truncexpon" and lib == "scipy":
        args = args[1:]

    if dist == "voigt" and lib == "scipy":
        from scipy.special import voigt_profile as method
    else:
        if lib == "scipy":
            d = getattr(scipy.stats, dist)
        else:
            d = importlib.import_module(f"numba_stats.{dist}")
        method = getattr(d, meth)
        if lib == "ours:parallel,fastmath":
            orig = method

            if len(args) == 2:

                @nb.njit(parallel=True, fastmath=True)
                def method(x, a, b):
                    return orig(x, a, b)

            elif len(args) == 3:

                @nb.njit(parallel=True, fastmath=True)
                def method(x, a, b, c):
                    return orig(x, a, b, c)

            elif len(args) == 4:

                @nb.njit(parallel=True, fastmath=True)
                def method(x, a, b, c, d):
                    return orig(x, a, b, c, d)

            else:
                raise AssertionError

    # warm-up JIT
    method(x, *args)
    benchmark(method, x, *args)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("lib", ("scipy", "ours", "ours:parallel,fastmath"))
def test_speed_voigt_pdf(benchmark, lib, n):
    x = np.linspace(-5, 5, n)
    rng = np.random.default_rng(1)
    rng.shuffle(x)

    sigma = 1.2
    gamma = 0.5

    if lib == "scipy":
        from scipy.special import voigt_profile

        def method(x, gamma, loc, sigma):
            return voigt_profile(x - loc, sigma, gamma)

    else:
        from numba_stats import voigt

        if lib == "ours:parallel,fastmath":

            @nb.njit(parallel=True, fastmath=True)
            def method(x, gamma, loc, sigma):
                return voigt.pdf(x, gamma, loc, sigma)

        else:
            method = voigt.pdf

    # warm-up JIT
    method(x, gamma, 0.0, sigma)
    benchmark(method, x, gamma, 0.0, sigma)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("lib", ("scipy", "ours", "ours:parallel,fastmath"))
def test_speed_bernstein_density(benchmark, lib, n):
    from scipy.interpolate import BPoly

    from numba_stats import bernstein

    x = np.linspace(0, 1, n)
    rng = np.random.default_rng(1)
    rng.shuffle(x)

    beta = np.arange(1, 4, dtype=float)
    xmin = np.min(x)
    xmax = np.max(x)

    if lib == "scipy":

        def method(x, beta, xmin, xmax):
            return BPoly(np.array(beta)[:, np.newaxis], [xmin, xmax])(x)

    else:
        method = bernstein.density

        if lib == "ours:parallel,fastmath":

            @nb.njit(parallel=True, fastmath=True)
            def method(x, gamma, loc, sigma):
                return bernstein.density(x, gamma, loc, sigma)

    # warm-up JIT
    method(x, beta, xmin, xmax)
    benchmark(method, x, beta, xmin, xmax)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("lib", ("scipy", "ours", "ours:parallel,fastmath"))
def test_speed_truncexpon_pdf_plus_norm_pdf(benchmark, lib, n):
    x = np.linspace(0, 1, n)
    rng = np.random.default_rng(1)
    rng.shuffle(x)

    xmin = np.min(x)
    xmax = np.max(x)

    if lib == "scipy":
        from scipy.stats import norm, truncexpon

        def method(x, z, mu, sigma, slope):
            p1 = truncexpon.pdf(x, xmax, xmin, slope)
            p2 = norm.pdf(x, mu, sigma)
            return (1 - z) * p1 + z * p2

    else:
        from numba_stats import norm, truncexpon

        def method(x, z, mu, sigma, slope):
            p1 = truncexpon.pdf(x, xmin, xmax, 0.0, slope)
            p2 = norm.pdf(x, mu, sigma)
            return (1 - z) * p1 + z * p2

        if lib == "ours:parallel,fastmath":
            method = nb.njit(parallel=True, fastmath=True)(method)

    # warm-up JIT
    args = 0.5, 0.5, 0.1, 1.0
    method(x, *args)
    benchmark(method, x, *args)
