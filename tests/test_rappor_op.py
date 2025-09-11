import os
import sys
import numpy as np
import mmh3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dpest.operations import rappor_distribution, one_time_rappor_distribution


def _hashed_indices(val, n_hashes, filter_size):
    return [mmh3.hash(str(val), seed=i) % filter_size for i in range(n_hashes)]


def test_one_time_rappor_distribution():
    val = 123
    n_hashes = 2
    filter_size = 8
    f = 0.6

    dists = one_time_rappor_distribution(val, n_hashes=n_hashes, filter_size=filter_size, f=f)
    hashed = _hashed_indices(val, n_hashes, filter_size)

    for i, dist in enumerate(dists):
        atoms = dict(dist.atoms)
        prob1 = 1 - f / 2 if i in hashed else f / 2
        assert np.isclose(atoms.get(1.0, 0.0), prob1)
        assert np.isclose(atoms.get(0.0, 0.0), 1 - prob1)


def test_rappor_distribution():
    val = 456
    n_hashes = 2
    filter_size = 8
    f = 0.6
    p = 0.3
    q = 0.7

    dists = rappor_distribution(val, n_hashes=n_hashes, filter_size=filter_size, f=f, p=p, q=q)
    hashed = _hashed_indices(val, n_hashes, filter_size)

    for i, dist in enumerate(dists):
        atoms = dict(dist.atoms)
        if i in hashed:
            prob1 = q * (1 - f / 2) + p * (f / 2)
        else:
            prob1 = q * (f / 2) + p * (1 - f / 2)
        assert np.isclose(atoms.get(1.0, 0.0), prob1)
        assert np.isclose(atoms.get(0.0, 0.0), 1 - prob1)
