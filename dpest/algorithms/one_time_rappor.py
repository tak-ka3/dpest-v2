"""One-time RAPPOR mechanism."""

from typing import List

import numpy as np

try:
    import mmh3  # optional dependency
except ModuleNotFoundError:  # pragma: no cover - optional
    mmh3 = None

from ..core import Dist
from ..operations import Condition
from ._helpers import expect_single_value
from .registry import auto_dist


@auto_dist()
def one_time_rappor(
    values: List[Dist],
    eps: float,
    n_hashes: int = 4,
    filter_size: int = 20,
    f: float = 0.95,
) -> List[Dist]:
    if mmh3 is None:  # pragma: no cover - optional
        raise ModuleNotFoundError("mmh3 is required for RAPPOR distributions")

    val = int(expect_single_value(values, "one_time_rappor"))
    filter_bits = np.zeros(filter_size, dtype=int)
    for i in range(n_hashes):
        idx = mmh3.hash(str(val), seed=i) % filter_size
        filter_bits[idx] = 1

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
