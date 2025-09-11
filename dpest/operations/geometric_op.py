"""Analytic distribution for the Truncated Geometric mechanism."""

from typing import List, Tuple

from ..core import Dist
from ..mechanisms.geometric import TruncatedGeometricMechanism


class TruncatedGeometric:
    """Operation producing the exact distribution of ``GeoSample``.

    Unlike other mechanisms in this package, the Truncated Geometric cannot be
    easily expressed as a combination of basic operations such as :class:`Add`
    and :class:`Condition`.  The algorithm requires handling an intermediate
    uniform distribution over a space of size ``(2^{k+1}+1)(2^k+1)^{n-1}``,
    which becomes prohibitively large even for modest ``n``.  Therefore we
    provide a direct analytic computation of the output distribution instead of
    composing lower level operations.
    """

    @staticmethod
    def apply(c: int, eps: float = 0.1, n: int = 5) -> Dist:
        """Return the exact output distribution of the mechanism.

        Args:
            c: result of the counting query, must satisfy ``0 <= c <= n``
            eps: privacy parameter epsilon
            n: number of individuals subject to the counting query

        Returns:
            ``Dist`` representing the probability mass function over
            ``{0, ..., n}``
        """
        mech = TruncatedGeometricMechanism(eps=eps, n=n)
        f = mech._compute_f(c)
        d = mech.d

        atoms: List[Tuple[float, float]] = []
        prev = 0
        for z in range(n + 1):
            prob = (f[z] - prev) / d
            atoms.append((float(z), prob))
            prev = f[z]

        return Dist.from_atoms(atoms)


def truncated_geometric_distribution(c: int, eps: float = 0.1, n: int = 5) -> Dist:
    """Convenience wrapper around :meth:`TruncatedGeometric.apply`."""

    return TruncatedGeometric.apply(c, eps=eps, n=n)
