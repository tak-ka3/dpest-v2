"""Tests for truncated_geometric algorithm."""

import pytest
import numpy as np
from dpest.algorithms import truncated_geometric, truncated_geometric_dist
from dpest.core import Dist


def test_truncated_geometric_basic():
    """Test basic functionality of truncated_geometric."""
    # c=2, eps=0.1, n=5
    # Dist.deterministic()を使って確定値を作成
    c_dist = Dist.deterministic(2.0)
    result = truncated_geometric([c_dist], eps=0.1, n=5)

    # 結果はDist型
    assert isinstance(result, Dist)

    # 結果は {0, 1, 2, 3, 4, 5} 上の分布
    assert len(result.atoms) == 6

    # 値は0から5まで（順序は問わない）
    values = sorted([int(v) for v, _ in result.atoms])
    assert values == [0, 1, 2, 3, 4, 5]

    # 確率の合計は1
    total_prob = sum(w for _, w in result.atoms)
    assert abs(total_prob - 1.0) < 1e-10


def test_truncated_geometric_dist_wrapper():
    """Test auto-generated dist wrapper function."""
    # numpy配列を渡せる
    result = truncated_geometric_dist(np.array([2]), eps=0.1, n=5)

    # 結果はDist型
    assert isinstance(result, Dist)

    # 結果は {0, 1, 2, 3, 4, 5} 上の分布
    assert len(result.atoms) == 6


def test_truncated_geometric_edge_cases():
    """Test edge cases."""
    # c=0 (最小値)
    c_dist = Dist.deterministic(0.0)
    result = truncated_geometric([c_dist], eps=0.1, n=5)
    assert len(result.atoms) == 6
    total_prob = sum(w for _, w in result.atoms)
    assert abs(total_prob - 1.0) < 1e-10

    # c=5 (最大値)
    c_dist = Dist.deterministic(5.0)
    result = truncated_geometric([c_dist], eps=0.1, n=5)
    assert len(result.atoms) == 6
    total_prob = sum(w for _, w in result.atoms)
    assert abs(total_prob - 1.0) < 1e-10


def test_truncated_geometric_invalid_input():
    """Test that invalid inputs raise errors."""
    # 複数の値を渡すとエラー
    dists = [Dist.deterministic(float(x)) for x in [1, 2, 3]]
    with pytest.raises(ValueError, match="single value"):
        truncated_geometric(dists, eps=0.1, n=5)


def test_truncated_geometric_probability_distribution():
    """Test that the probability distribution is reasonable."""
    c_dist = Dist.deterministic(2.0)
    result = truncated_geometric([c_dist], eps=0.1, n=5)

    # c=2の場合、出力が2に近い値になる確率が高いはず
    # atoms は [(value, weight), ...] の形式
    prob_dict = {int(v): w for v, w in result.atoms}

    # 確率はすべて非負
    for prob in prob_dict.values():
        assert prob >= 0

    # 確率の合計は1
    assert abs(sum(prob_dict.values()) - 1.0) < 1e-10


def test_truncated_geometric_with_distribution_input():
    """Test that algorithm works with probabilistic input using mux operation."""
    # c が確率分布の場合をテスト
    # 例: c が 50% の確率で 1、50% の確率で 2
    c_dist = Dist.from_atoms([(1.0, 0.5), (2.0, 0.5)])
    result = truncated_geometric([c_dist], eps=0.1, n=5)

    # 結果は {0, 1, 2, 3, 4, 5} 上の分布のはず
    assert isinstance(result, Dist)

    # 確率の合計は1
    total_prob = sum(w for _, w in result.atoms)
    assert abs(total_prob - 1.0) < 1e-10

    # 確率はすべて非負
    for _, prob in result.atoms:
        assert prob >= 0

    # c=1 と c=2 の分布の混合になっているはず
    # （詳細な検証は省略するが、基本的な整合性をチェック）
    prob_dict = {v: w for v, w in result.atoms}
    # すべての出力値は [0, n] の範囲内
    for val in prob_dict.keys():
        assert 0 <= val <= 5
