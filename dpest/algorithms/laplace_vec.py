"""Laplace mechanism applied element-wise to a vector."""

from typing import List

from ..core import Dist
from ..operations import vector_add
from ..noise import create_laplace_noise
from .registry import auto_dist


@auto_dist()
def laplace_vec(values: List[Dist], eps: float) -> List[Dist]:
    """Adds Laplace noise (scale 1/eps) to each element in the vector."""
    noise_dists = create_laplace_noise(b=1 / eps, size=len(values))
    return vector_add(values, noise_dists)
