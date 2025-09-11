"""Analytic distributions for RAPPOR mechanisms using operation compositions."""
from typing import List
import numpy as np
import mmh3
from ..core import Dist
from .condition_op import Condition

class OneTimeRappor:
    """One-time RAPPOR (steps 1-2) using analytic operations.

    Returns a list of distributions, one for each bit in the Bloom filter.
    """

    @staticmethod
    def apply(val, n_hashes: int = 4, filter_size: int = 20, f: float = 0.95) -> List[Dist]:
        # Populate bloom filter deterministically
        filter_bits = np.zeros(filter_size, dtype=int)
        for i in range(n_hashes):
            idx = mmh3.hash(str(val), seed=i) % filter_size
            filter_bits[idx] = 1

        # Pre-computed distributions used in the permanent randomized response
        cond_randomize = Dist.from_atoms([(1.0, f), (0.0, 1.0 - f)])
        cond_flip = Dist.from_atoms([(1.0, 0.5), (0.0, 0.5)])
        bit_one = Dist.deterministic(1.0)
        bit_zero = Dist.deterministic(0.0)
        random_bit = Condition.apply(cond_flip, bit_one, bit_zero)

        dists: List[Dist] = []
        for bit in filter_bits:
            base = Dist.deterministic(float(bit))
            perm = Condition.apply(cond_randomize, random_bit, base)
            dists.append(perm)
        return dists


def one_time_rappor_distribution(val, n_hashes: int = 4, filter_size: int = 20, f: float = 0.95) -> List[Dist]:
    """Convenience function for OneTimeRappor.apply"""
    return OneTimeRappor.apply(val, n_hashes=n_hashes, filter_size=filter_size, f=f)


class Rappor:
    """Full RAPPOR mechanism (steps 1-3) using analytic operations."""

    @staticmethod
    def apply(val, n_hashes: int = 4, filter_size: int = 20,
              f: float = 0.75, p: float = 0.45, q: float = 0.55) -> List[Dist]:
        # Permanent randomized response using OneTimeRappor
        perm_dists = OneTimeRappor.apply(val, n_hashes=n_hashes,
                                         filter_size=filter_size, f=f)

        # Instantaneous randomized response distributions
        dist_if_one = Dist.from_atoms([(1.0, q), (0.0, 1.0 - q)])
        dist_if_zero = Dist.from_atoms([(1.0, p), (0.0, 1.0 - p)])

        dists: List[Dist] = []
        for perm in perm_dists:
            final = Condition.apply(perm, dist_if_one, dist_if_zero)
            dists.append(final)
        return dists


def rappor_distribution(val, n_hashes: int = 4, filter_size: int = 20,
                        f: float = 0.75, p: float = 0.45, q: float = 0.55) -> List[Dist]:
    """Convenience function for Rappor.apply"""
    return Rappor.apply(val, n_hashes=n_hashes, filter_size=filter_size,
                         f=f, p=p, q=q)
