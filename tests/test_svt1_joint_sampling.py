import math
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dpest.engine import compile
from dpest.algorithms.svt1 import svt1
from dpest.core import Dist


def test_svt1_sampling_fallback_preserves_abort():
    # deterministic queries that will likely exceed threshold quickly
    queries = [5.0, 5.0, 5.0, 5.0]
    compiled = compile(lambda q: svt1(q, eps=0.5, t=0.0, c=1))
    result = compiled(queries)

    joint = result[0]._joint_samples
    assert joint.shape[1] == len(queries)

    for row in joint:
        seen_nan = False
        for value in row:
            if isinstance(value, float) and math.isnan(value):
                seen_nan = True
            elif seen_nan:
                assert math.isnan(value)

    # 2つ目以降の分布には NaN の原子が含まれるはず
    second = result[1]
    assert any(isinstance(v, float) and math.isnan(v) for v, _ in second.atoms)
