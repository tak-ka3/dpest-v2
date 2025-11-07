"""Report Noisy Max variant 4 (exponential noise, max value)."""

from typing import List

from ..core import Dist
from ..engine import vector_add, vector_max
from ..noise import create_exponential_noise
from .registry import auto_dist


@auto_dist()
def report_noisy_max4(values: List[Dist], eps: float) -> Dist:
    """Adds exponential noise (scale 2/eps) and returns the noisy maximum value."""
    noise_dists = create_exponential_noise(b=2 / eps, size=len(values))
    return vector_max(vector_add(values, noise_dists))
