"""Laplace mechanism applied in parallel."""

from typing import List

from ..core import Dist
from ..noise import create_laplace_noise
from ..operations import add
from ._helpers import expect_single_value
from .registry import auto_dist


@auto_dist()
def laplace_parallel(values: List[Dist], eps_each: float, n_parallel: int) -> List[Dist]:
    """Adds independent Laplace noise to repeated copies of the scalar input."""
    scalar = expect_single_value(values, "laplace_parallel")
    base = Dist.deterministic(scalar)
    noise_list = create_laplace_noise(b=1 / eps_each, size=n_parallel)
    return [add(base, n) for n in noise_list]
