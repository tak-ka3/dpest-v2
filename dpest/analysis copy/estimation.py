"""
汎用的な差分プライバシー推定関数。

examples/ 以下でも再利用されるが、ライブラリ側に配置しておくことで
テストや他モジュールからも一貫して利用できる。
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core import Dist
from ..engine import set_sampling_samples
from ..utils.input_patterns import generate_patterns
from ..utils.privacy import (
    epsilon_from_dist,
    epsilon_from_list_joint,
    epsilon_from_samples_matrix,
)


def estimate_algorithm(
    name: str,
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    dist_func: Optional[Callable[..., Sequence[Dist] | Dist]] = None,
    joint_dist_func: Optional[Callable[..., Dist | List[Dist]]] = None,
    mechanism=None,
    eps: float = 0.1,
    n_samples: int = 100_000,
    extra: Optional[Iterable] = None,
    verbose: bool = False,
    hist_bins: int = 100,
) -> float:
    """
    指定したアルゴリズムについてプライバシー損失 ε を推定する。

    Args:
        name: アルゴリズム名（ログ用）
        pairs: 隣接データセット (D, D') のタプル列
        dist_func: 各入力に対して出力分布 (Dist または List[Dist]) を返す関数
        joint_dist_func: ジョイント分布を直接返す関数がある場合はこちらを優先
        mechanism: サンプリングのみ提供する Mechanism 実装を指定する場合
        eps: アルゴリズムに渡すプライバシーパラメータ
        n_samples: mechanism ベースで推定する際のサンプル数
        extra: ラッパー関数に追加で渡すパラメータ
        verbose: True の場合、利用した推定経路を標準出力へログ
    """

    eps_max = 0.0
    # エンジンのサンプリング回数も外部から設定する
    set_sampling_samples(n_samples)
    if verbose:
        print(
            f"[estimate_algorithm] name={name}, "
            f"joint_dist_func={joint_dist_func}, dist_func={dist_func}, mechanism={mechanism}"
        )

    for D, Dp in pairs:
        if joint_dist_func is not None:
            args = (D, eps) if extra is None else (D, eps, *extra)
            P = joint_dist_func(*args)
            args_prime = (Dp, eps) if extra is None else (Dp, eps, *extra)
            Q = joint_dist_func(*args_prime)
            eps_val = epsilon_from_dist(P, Q)
        elif dist_func is not None:
            args = (D, eps) if extra is None else (D, eps, *extra)
            P = dist_func(*args)
            args_prime = (Dp, eps) if extra is None else (Dp, eps, *extra)
            Q = dist_func(*args_prime)
            if isinstance(P, list):
                eps_val = epsilon_from_list_joint(P, Q, bins=hist_bins)
            else:
                eps_val = epsilon_from_dist(P, Q)
        else:
            P_samples = mechanism.m(D, n_samples)
            Q_samples = mechanism.m(Dp, n_samples)
            eps_val = epsilon_from_samples_matrix(P_samples, Q_samples, bins=hist_bins)

        eps_max = max(eps_max, eps_val)

    return eps_max


def generate_hist_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """指定長のヒストグラム用隣接ペアを生成。"""

    return list(generate_patterns(length).values())


def generate_change_one_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """単純な change-one パターンを生成（現在は generate_patterns と同じ）。"""

    return list(generate_patterns(length).values())
