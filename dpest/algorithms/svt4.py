"""
SVT4 (Sparse Vector Technique 4) - 分布ベース実装

Alg. 4 from:
    M. Lyu, D. Su, and N. Li. 2017.
    Understanding the Sparse Vector Technique for Differential Privacy.
    Proceedings of the VLDB Endowment.

SVT4の特徴: 各クエリにノイズを追加（eps1=eps/4, eps2=eps-eps1, scale=1/eps2）
"""

from typing import List
from ..core import Dist
from ..noise import Laplace
from ..operations import add, affine, mux, geq
from .registry import auto_dist

# NANセンチネル値
NAN = float('nan')


@auto_dist()
def svt4(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 1) -> List[Dist]:
    """
    SVT4アルゴリズムの分布ベース実装

    Args:
        queries: クエリ結果の分布のリスト
        eps: プライバシーパラメータ（デフォルト: 0.1）
        t: 閾値（デフォルト: 1.0）
        c: 最大出力回数（cutoff、デフォルト: 1）

    Returns:
        各クエリに対する出力分布のリスト
        各分布は {0: FALSE, 1: TRUE, NAN: ABORTED} の確率質量を持つ
    """
    # プライバシーパラメータの分割（SVT4特有）
    eps1 = eps / 4.0
    eps2 = eps - eps1

    # 閾値にノイズを追加: T = t + Laplace(b=1/eps1)
    lap_T = Laplace(b=1/eps1).to_dist()
    T = affine(lap_T, 1.0, t)

    # カウンタと打ち切りフラグ
    count = Dist.deterministic(0.0)
    broken = Dist.deterministic(0.0)
    result = []

    for Q in queries:
        # クエリにノイズを追加（スケールが異なる）
        lap_Q = Laplace(b=1/eps2).to_dist()
        noisy_Q = add(Q, lap_Q)

        # 閾値と比較
        over = geq(noisy_Q, T)

        # 打ち切り後はNANを出力
        out_i = mux(broken, NAN, over)
        result.append(out_i)

        # カウンタを更新（打ち切り後は加算しない）
        inc = mux(broken, 0, over)
        count = add(count, inc)
        broken = geq(count, c)

    return result
