"""
SVT6 (Sparse Vector Technique 6) - 分布ベース実装

Alg. 6 from:
    M. Lyu, D. Su, and N. Li. 2017.
    Understanding the Sparse Vector Technique for Differential Privacy.
    Proceedings of the VLDB Endowment.

SVT6の特徴: 各クエリにノイズを追加、カウンタや打ち切りなし
"""

from typing import List
from ..core import Dist
from ..noise import Laplace
from ..operations import add, affine, geq
from .registry import auto_dist


@auto_dist()
def svt6(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 1) -> List[Dist]:
    """
    SVT6アルゴリズムの分布ベース実装

    Args:
        queries: クエリ結果の分布のリスト
        eps: プライバシーパラメータ（デフォルト: 0.1）
        t: 閾値（デフォルト: 1.0）
        c: 最大出力回数（未使用、互換性のため）

    Returns:
        各クエリに対する出力分布のリスト
        各分布は {0: FALSE, 1: TRUE} の確率質量を持つ
    """
    # プライバシーパラメータの分割
    eps1 = eps / 2.0
    eps2 = eps - eps1

    # 閾値にノイズを追加: T = t + Laplace(b=1/eps1)
    lap_T = Laplace(b=1/eps1).to_dist()
    T = affine(lap_T, 1.0, t)

    # カウンタや打ち切りなし、各クエリを独立に処理
    result = []
    for Q in queries:
        # クエリにノイズを追加
        lap_Q = Laplace(b=1/eps2).to_dist()
        noisy_Q = add(Q, lap_Q)

        # 閾値と比較
        over = geq(noisy_Q, T)
        result.append(over)

    return result
