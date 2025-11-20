"""Test truncated geometric algorithm implementation."""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dpest.algorithms import truncated_geometric, truncated_geometric_dist
from dpest.mechanisms.geometric import TruncatedGeometricMechanism
from dpest.core import Dist


def test_truncated_geometric_distribution():
    """Test that algorithm implementation matches mechanism."""
    c = 2
    eps = 0.1
    n = 5

    # Algorithm implementation (using Dist operations)
    c_dist = Dist.deterministic(float(c))
    dist_algo = truncated_geometric([c_dist], eps=eps, n=n)

    # Also test the dist wrapper
    dist_wrapper = truncated_geometric_dist(np.array([c]), eps=eps, n=n)

    # Reference implementation (mechanism)
    mech = TruncatedGeometricMechanism(eps=eps, n=n)
    f = mech._compute_f(c)
    d = mech.d

    # Expected probabilities from mechanism
    expected = {}
    prev = 0
    for z in range(n + 1):
        prob = (f[z] - prev) / d
        expected[float(z)] = prob
        prev = f[z]

    # Verify algorithm implementation matches expected
    atoms_algo = dict(dist_algo.atoms)
    for z, p in expected.items():
        assert np.isclose(atoms_algo.get(z, 0.0), p, atol=1e-4), f"Mismatch at z={z}: {atoms_algo.get(z, 0.0)} != {p}"
    assert np.isclose(dist_algo.total_mass(), 1.0, atol=1e-4)

    # Verify wrapper matches algorithm
    atoms_wrapper = dict(dist_wrapper.atoms)
    for z, p in expected.items():
        assert np.isclose(atoms_wrapper.get(z, 0.0), p, atol=1e-4), f"Wrapper mismatch at z={z}"
