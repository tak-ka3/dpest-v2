from typing import List
import numpy as np

from ..core import Dist
from ..noise import create_laplace_noise
from .operations import add_distributions
from .condition_op import compare_geq


class SVT5:
    """Operation computing analytic distributions for Sparse Vector Technique 5."""

    @staticmethod
    def apply(a, eps: float = 0.1, t: float = 1.0) -> List[Dist]:
        """Return per-query output distributions for SVT5.

        Args:
            a: array of query results (sensitivity 1)
            eps: privacy parameter epsilon
            t: threshold value

        Returns:
            list of `Dist` objects representing the probability of TRUE(1)
            or FALSE(0) for each query.
        """
        x = np.atleast_1d(a)
        eps1 = eps / 2.0
        rho_dist = create_laplace_noise(b=1 / eps1)
        thresh_dist = add_distributions(rho_dist, Dist.deterministic(t))
        results: List[Dist] = []
        for val in x:
            val_dist = Dist.deterministic(float(val))
            res_dist = compare_geq(val_dist, thresh_dist)
            results.append(res_dist)
        return results


def svt5_distribution(a, eps: float = 0.1, t: float = 1.0) -> List[Dist]:
    """Convenience wrapper for :class:`SVT5`."""
    return SVT5.apply(a, eps=eps, t=t)
