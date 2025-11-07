"""Report Noisy Max variant 3 (Laplace noise, max value)."""

from typing import List

from ..core import Dist
from ..engine import vector_add, vector_max
from ..noise import create_laplace_noise
from .registry import auto_dist


@auto_dist()
def report_noisy_max3(values: List[Dist], eps: float) -> Dist:
    """Adds Laplace noise (scale 2/eps) and returns the noisy maximum value."""
    noise_dists = create_laplace_noise(b=2 / eps, size=len(values))
    return vector_max(vector_add(values, noise_dists))
