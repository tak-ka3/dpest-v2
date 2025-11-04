"""
SVT1 (Sparse Vector Technique 1) - 分布ベース実装

Alg. 1 from:
    M. Lyu, D. Su, and N. Li. 2017.
    Understanding the Sparse Vector Technique for Differential Privacy.
    Proceedings of the VLDB Endowment.

この実装は確率分布を直接計算し、compile()で最適化された分布計算関数を生成します。
"""

from typing import List
import math
import numpy as np
from ..core import Dist
from ..engine import FallbackResult
from ..noise import Laplace
from ..operations import Add, Affine, mux, compare_geq as GE

# NANセンチネル値
NAN = float('nan')


def svt1(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]:
    """
    SVT1アルゴリズムの分布ベース実装

    Args:
        queries: クエリ結果の分布のリスト
        eps: プライバシーパラメータ（デフォルト: 0.1）
        t: 閾値（デフォルト: 1.0）
        c: 最大出力回数（cutoff、デフォルト: 2）

    Returns:
        各クエリに対する出力分布のリスト
        各分布は {0: FALSE, 1: TRUE, NAN: ABORTED} の確率質量を持つ
    """
    # プライバシーパラメータの分割
    eps1 = eps / 2.0
    eps2 = eps - eps1

    # 閾値にノイズを追加: T = t + Laplace(b=1/eps1)
    lap_T = Laplace(b=1/eps1).to_dist()
    T = Affine.apply(lap_T, 1.0, t)

    # カウンタと打ち切りフラグ
    count = Dist.deterministic(0.0)
    broken = Dist.deterministic(0.0)
    result = []

    for Q in queries:
        # クエリにノイズを追加
        lap_Q = Laplace(b=2*c/eps2).to_dist()
        noisy_Q = Add.apply(Q, lap_Q)

        # 閾値と比較
        over = GE(noisy_Q, T)

        # 打ち切り後はNANを出力
        out_i = mux(broken, NAN, over)
        result.append(out_i)

        # カウンタを更新（打ち切り後は加算しない）
        inc = mux(broken, 0, over)
        count = Add.apply(count, inc)
        broken = GE(count, c)

    return FallbackResult(
        result,
        lambda raw_input, n_samples: _svt1_sampler(raw_input, eps, t, c, n_samples),
    )


def _to_numeric_queries(raw_input: List[Dist], eps: float):
    values = []
    if isinstance(raw_input, np.ndarray):
        iterable = raw_input.tolist()
    elif isinstance(raw_input, (list, tuple)):
        iterable = raw_input
    else:
        iterable = [raw_input]

    for item in iterable:
        if isinstance(item, Dist):
            # deterministicのみサポート
            if len(item.atoms) == 1 and math.isclose(item.atoms[0][1], 1.0, rel_tol=1e-9):
                values.append(float(item.atoms[0][0]))
            else:
                raise ValueError("Sampling fallback requires deterministic Dist inputs")
        else:
            values.append(float(item))

    return np.array(values, dtype=float)


def _svt1_sampler(raw_input, eps: float, t: float, c: int, n_samples: int):
    values = _to_numeric_queries(raw_input, eps)
    num_queries = len(values)
    if num_queries == 0:
        return np.empty((n_samples, 0))

    eps1 = eps / 2.0
    eps2 = eps - eps1
    thresh_scale = 1 / eps1
    query_scale = 2 * c / eps2

    samples = np.empty((n_samples, num_queries), dtype=float)

    for i in range(n_samples):
        rho = np.random.laplace(loc=0.0, scale=thresh_scale)
        threshold = t + rho
        count = 0
        aborted = False

        for j, val in enumerate(values):
            if aborted:
                samples[i, j] = np.nan
                continue

            noisy_val = val + np.random.laplace(loc=0.0, scale=query_scale)
            if noisy_val >= threshold:
                samples[i, j] = 1.0
                count += 1
                if count >= c:
                    aborted = True
            else:
                samples[i, j] = 0.0

        if aborted:
            for k in range(j + 1, num_queries):
                samples[i, k] = np.nan

    return samples
