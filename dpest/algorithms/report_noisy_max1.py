"""Report Noisy Max variant 1."""

from typing import List

from ..core import Dist
from ..engine import vector_add, vector_argmax
from ..noise import create_laplace_noise
from .registry import auto_dist


@auto_dist()
def report_noisy_max1(values: List[Dist], eps: float) -> Dist:
    """Adds Laplace noise (scale 2/eps) and returns argmax index distribution."""
    noise_dists = create_laplace_noise(b=2 / eps, size=len(values))
    return vector_argmax(vector_add(values, noise_dists))
