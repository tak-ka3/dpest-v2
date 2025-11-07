import warnings

import numpy as np
import pytest

from dpest.core import Dist
from dpest.engine import compile
from dpest.algorithms import report_noisy_max1_dist
from dpest.algorithms.svt1 import svt1


def test_report_noisy_max1_dist_produces_distribution():
    data = np.array([0.0, 1.0, 2.0], dtype=float)
    dist = report_noisy_max1_dist(data, eps=0.5)

    assert isinstance(dist, Dist)
    assert dist.total_mass() == pytest.approx(1.0, rel=1e-6)


def test_svt1_compilation_and_execution():
    queries = [Dist.deterministic(float(i)) for i in range(3)]
    compiled = compile(lambda q: svt1(q, eps=0.3, t=0.0, c=1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compiled(queries)

    assert isinstance(result, list)
    assert len(result) == len(queries)
    for dist in result:
        assert isinstance(dist, Dist)
        assert dist.total_mass() == pytest.approx(1.0, rel=1e-6)
