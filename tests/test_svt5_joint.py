import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from examples.privacy_loss_report import (
    estimate_algorithm,
    svt5_joint_dist,
    generate_change_one_pairs,
)


def test_svt5_joint_runs():
    pairs = generate_change_one_pairs(3)
    eps = 0.1
    # 単にジョイント分布による推定が実行できることを確認する
    joint = estimate_algorithm(
        "SVT5", pairs, joint_dist_func=svt5_joint_dist, eps=eps
    )
    assert joint >= 0.0
