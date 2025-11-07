"""RAPPOR mechanism built on top of one-time RAPPOR."""

from typing import List

try:
    import mmh3  # optional dependency
except ModuleNotFoundError:  # pragma: no cover - optional
    mmh3 = None

from ..core import Dist
from ..operations import Condition
from .one_time_rappor import one_time_rappor
from .registry import auto_dist


@auto_dist()
def rappor(
    values: List[Dist],
    eps: float,
    n_hashes: int = 4,
    filter_size: int = 20,
    f: float = 0.75,
    p: float = 0.45,
    q: float = 0.55,
) -> List[Dist]:
    if mmh3 is None:  # pragma: no cover - optional
        raise ModuleNotFoundError("mmh3 is required for RAPPOR distributions")

    perm_dists = one_time_rappor(values, eps, n_hashes=n_hashes, filter_size=filter_size, f=f)
    dist_if_one = Dist.from_atoms([(1.0, q), (0.0, 1.0 - q)])
    dist_if_zero = Dist.from_atoms([(1.0, p), (0.0, 1.0 - p)])
    dists: List[Dist] = []
    for perm in perm_dists:
        final = Condition.apply(perm, dist_if_one, dist_if_zero)
        dists.append(final)
    return dists
