# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from typing import Any

from numba.extending import get_cython_function_address
from numba.types import WrapperAddressProtocol, float64


def get(name: str, signature: Any) -> Any:
    # create new function object with correct signature that numba can call
    from scipy.special import cython_special

    # scipy-1.12 started to provide fused versions for some special functions
    if name == "betainc":
        fuse_name = f"__pyx_fuse_0{name}"
    else:
        fuse_name = f"__pyx_fuse_1{name}"
    if fuse_name not in cython_special.__pyx_capi__:
        fuse_name = name

    addr = get_cython_function_address("scipy.special.cython_special", fuse_name)

    # dynamically create type that inherits from WrapperAddressProtocol
    cls = type(
        name,
        (WrapperAddressProtocol,),
        {"__wrapper_address__": lambda self: addr, "signature": lambda self: signature},
    )
    return cls()


# unary functions (double)
ndtri = get("ndtri", float64(float64))

# binary functions (double)
gammainc = get("gammainc", float64(float64, float64))
gammaincc = get("gammaincc", float64(float64, float64))
stdtrit = get("stdtrit", float64(float64, float64))
stdtr = get("stdtr", float64(float64, float64))

# n-ary functions (double)
voigt_profile = get("voigt_profile", float64(float64, float64, float64))
xlogy = get("xlogy", float64(float64, float64))
xlog1py = get("xlog1py", float64(float64, float64))
betainc = get("betainc", float64(float64, float64, float64))
