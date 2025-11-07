"""Noisy histogram mechanism (variant 1)."""

from typing import List

from ..core import Dist
from ..engine import vector_add
from ..noise import create_laplace_noise
from .registry import auto_dist


@auto_dist()
def noisy_hist1(values: List[Dist], eps: float) -> List[Dist]:
    """Add Laplace noise with scale 1/eps to each histogram bucket."""
    noise_dists = create_laplace_noise(b=1 / eps, size=len(values))
    return vector_add(values, noise_dists)
