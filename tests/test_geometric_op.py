import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dpest.operations import TruncatedGeometric, truncated_geometric_distribution
from dpest.mechanisms.geometric import TruncatedGeometricMechanism


def test_truncated_geometric_distribution():
    c = 2
    eps = 0.1
    n = 5

    dist_func = truncated_geometric_distribution(c, eps=eps, n=n)
    dist_op = TruncatedGeometric.apply(c, eps=eps, n=n)

    mech = TruncatedGeometricMechanism(eps=eps, n=n)
    f = mech._compute_f(c)
    d = mech.d

    expected = {}
    prev = 0
    for z in range(n + 1):
        prob = (f[z] - prev) / d
        expected[float(z)] = prob
        prev = f[z]

    atoms = dict(dist_func.atoms)
    for z, p in expected.items():
        assert np.isclose(atoms.get(z, 0.0), p)
    assert np.isclose(dist_func.total_mass(), 1.0)

    # Operation class should match the convenience wrapper
    assert dist_func.atoms == dist_op.atoms
