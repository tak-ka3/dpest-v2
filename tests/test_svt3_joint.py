import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from examples.privacy_loss_report import (
    estimate_algorithm,
    svt3_dist,
    svt3_joint_dist,
    generate_change_one_pairs,
)


def test_svt3_joint_less_than_sum():
    pairs = generate_change_one_pairs(3)
    eps = 0.1
    naive = estimate_algorithm(
        "SVT3", pairs, dist_func=svt3_dist, eps=eps, extra=(2, 1.0)
    )
    joint = estimate_algorithm(
        "SVT3", pairs, joint_dist_func=svt3_joint_dist, eps=eps, extra=(2, 1.0)
    )
    assert not math.isclose(joint, naive)

