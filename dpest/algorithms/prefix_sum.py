"""
PrefixSum algorithm using dpest operations.

Reference:
    Y. Wang, Z. Ding, G. Wang, D. Kifer, D. Zhang. 2019.
    Proving differential privacy with shadow execution.
    PLDI 2019.
"""

from typing import List

from ..core import Dist
from ..operations import add
from ..noise import create_laplace_noise
from .registry import auto_dist


@auto_dist()
def prefix_sum(values: List[Dist], eps: float) -> List[Dist]:
    """
    Computes prefix sum with Laplace noise added to each element.

    Algorithm:
    1. Add Laplace(1/eps) noise to each input value
    2. Compute prefix sums: output[i] = sum(noisy_values[0:i+1])

    Args:
        values: List of input distributions
        eps: Privacy parameter

    Returns:
        List of prefix sum distributions
    """
    n = len(values)

    # Step 1: Add Laplace noise to each value
    noise_dists = create_laplace_noise(b=1/eps, size=n)
    noisy_values = [add(values[i], noise_dists[i]) for i in range(n)]

    # Step 2: Compute prefix sums
    result = []
    partial_sum = noisy_values[0]
    result.append(partial_sum)

    for i in range(1, n):
        partial_sum = add(partial_sum, noisy_values[i])
        result.append(partial_sum)

    return result
