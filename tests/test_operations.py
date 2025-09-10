import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dpest.core import Dist
from dpest.operations import add_distributions, max_distribution, min_distribution, argmax_distribution


def test_add_distribution():
    x = Dist.deterministic(1.0)
    y = Dist.deterministic(2.0)
    result = add_distributions(x, y)
    assert result.atoms == [(3.0, 1.0)]


def test_max_min_distribution():
    dists = [Dist.deterministic(v) for v in (1.0, 3.0, 2.0)]
    max_res = max_distribution(dists)
    min_res = min_distribution(dists)
    assert max_res.atoms == [(3.0, 1.0)]
    assert min_res.atoms == [(1.0, 1.0)]


def test_argmax_distribution():
    dists = [Dist.deterministic(v) for v in (1.0, 3.0, 2.0)]
    result = argmax_distribution(dists)
    probs = dict(result.atoms)
    assert probs.get(1) == 1.0
    assert probs.get(0, 0.0) == 0.0
    assert probs.get(2, 0.0) == 0.0
