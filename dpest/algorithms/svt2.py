"""
SVT2 (Sparse Vector Technique 2) - 分布ベース実装

Alg. 2 from:
    M. Lyu, D. Su, and N. Li. 2017.
    Understanding the Sparse Vector Technique for Differential Privacy.
    Proceedings of the VLDB Endowment.

SVT2の特徴: 閾値を各TRUEの後に再サンプリング
"""

from typing import List
from ..core import Dist
from ..noise import Laplace
from ..operations import add, affine, mux, geq
from .registry import auto_dist

# NANセンチネル値
NAN = float('nan')


@auto_dist()
def svt2(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]:
    """
    SVT2アルゴリズムの分布ベース実装

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

    # 初期閾値にノイズを追加: T = t + Laplace(b=c/eps1)
    lap_T = Laplace(b=c/eps1).to_dist()
    T = affine(lap_T, 1.0, t)

    # カウンタと打ち切りフラグ
    count = Dist.deterministic(0.0)
    broken = Dist.deterministic(0.0)
    result = []

    for Q in queries:
        # クエリにノイズを追加
        lap_Q = Laplace(b=2*c/eps2).to_dist()
        noisy_Q = add(Q, lap_Q)

        # 閾値と比較
        over = geq(noisy_Q, T)

        # 打ち切り後はNANを出力
        out_i = mux(broken, NAN, over)
        result.append(out_i)

        # TRUEの時に閾値を再サンプリング
        # 新しい閾値ノイズを生成
        new_lap_T = Laplace(b=c/eps1).to_dist()
        new_T = affine(new_lap_T, 1.0, t)
        # overが1（TRUE）の時だけ新しい閾値を使用
        T = mux(over, new_T, T)

        # カウンタを更新（打ち切り後は加算しない）
        inc = mux(broken, 0, over)
        count = add(count, inc)
        broken = geq(count, c)

    return result
