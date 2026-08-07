"""
We provide numba-accelerated implementations of common probability distributions.

* Uniform
* (Truncated) Normal
* Log-normal
* Poisson
* (Truncated) Exponential
* Student's t
* Voigtian
* Crystal Ball
* Generalised double-sided Crystal Ball
* Tsallis-Hagedorn, a model for the minimum bias pT distribution
* Q-Gaussian
* Bernstein density (not normalized to unity, use this in extended likelihood fits)
* Cruijff density (not normalized to unity, use this in extended likelihood fits)
* CMS-Shape
* Novosibirsk

The speed gains are huge, up to a factor of 100 compared to Scipy.

The distributions are optimized for the use in maximum-likelihood fits, where you query
a distribution at many points with a single set of parameters.
"""

from importlib.metadata import version

__version__ = version("numba-stats")
