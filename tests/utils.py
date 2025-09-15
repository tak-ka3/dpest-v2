from collections import defaultdict
from typing import List, Iterable, Tuple

from dpest.core import Dist
from dpest.utils.privacy import epsilon_from_list


def marginals_from_joint(dist: Dist) -> List[Dist]:
    """Return list of marginal distributions for each coordinate."""
    if not dist.atoms:
        return []
    length = len(dist.atoms[0][0])
    counts = [defaultdict(float) for _ in range(length)]
    for seq, p in dist.atoms:
        for i, val in enumerate(seq):
            counts[i][val] += p
    return [Dist.from_atoms(list(d.items())) for d in counts]


def naive_epsilon(
    pairs: Iterable[Tuple],
    eps: float,
    joint_func,
    extra=None,
) -> float:
    """Compute naive epsilon by summing marginal epsilons."""
    max_eps = 0.0
    for D, Dp in pairs:
        if extra is None:
            P = joint_func(D, eps)
            Q = joint_func(Dp, eps)
        else:
            P = joint_func(D, eps, *extra)
            Q = joint_func(Dp, eps, *extra)
        P_m = marginals_from_joint(P)
        Q_m = marginals_from_joint(Q)
        e = epsilon_from_list(P_m, Q_m)
        max_eps = max(max_eps, e)
    return max_eps
