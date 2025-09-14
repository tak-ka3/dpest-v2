import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from examples.privacy_loss_report import (
    estimate_algorithm,
    svt5_dist,
    svt5_joint_dist,
    generate_change_one_pairs,
)


def test_svt5_joint_less_than_sum():
    pairs = generate_change_one_pairs(3)
    eps = 0.1
    naive = estimate_algorithm(
        "SVT5", pairs, dist_func=svt5_dist, eps=eps
    )
    joint = estimate_algorithm(
        "SVT5", pairs, joint_dist_func=svt5_joint_dist, eps=eps
    )
    assert joint <= naive
    assert not math.isclose(joint, naive)
