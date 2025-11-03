"""
SVT5 (Sparse Vector Technique 5) - 分布ベース実装

Alg. 5 from:
    M. Lyu, D. Su, and N. Li. 2017.
    Understanding the Sparse Vector Technique for Differential Privacy.
    Proceedings of the VLDB Endowment.

SVT5の特徴: ノイズなしクエリと比較（非プライバシー保護、カウンタや打ち切りなし）
"""

from typing import List
from ..core import Dist
from ..noise import Laplace
from ..operations import Affine, compare_geq as GE


def svt5(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]:
    """
    SVT5アルゴリズムの分布ベース実装

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
    # eps2は使用されない（クエリにノイズを追加しない）

    # 閾値にノイズを追加: T = t + Laplace(b=1/eps1)
    lap_T = Laplace(b=1/eps1).to_dist()
    T = Affine.apply(lap_T, 1.0, t)

    # カウンタや打ち切りなし、各クエリを独立に比較
    result = []
    for Q in queries:
        # クエリにノイズを追加しない
        over = GE(Q, T)
        result.append(over)

    return result
