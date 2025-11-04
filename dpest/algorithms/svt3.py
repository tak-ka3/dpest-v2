"""
SVT3 (Sparse Vector Technique 3) - 分布ベース実装

Alg. 3 from:
    M. Lyu, D. Su, and N. Li. 2017.
    Understanding the Sparse Vector Technique for Differential Privacy.
    Proceedings of the VLDB Endowment.

SVT3の特徴: TRUEの時にノイズ付きクエリ値を出力、FALSEの時は-1000.0
"""

from typing import List
from ..core import Dist
from ..noise import Laplace
from ..operations import Add, Affine, mux, compare_geq as GE

# センチネル値
NAN = float('nan')
FALSE_SENTINEL = -1000.0  # FALSE出力用


def svt3(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]:
    """
    SVT3アルゴリズムの分布ベース実装

    Args:
        queries: クエリ結果の分布のリスト
        eps: プライバシーパラメータ（デフォルト: 0.1）
        t: 閾値（デフォルト: 1.0）
        c: 最大出力回数（cutoff、デフォルト: 2）

    Returns:
        各クエリに対する出力分布のリスト
        各分布は {ノイズ付き値: TRUE, -1000.0: FALSE, NAN: ABORTED} の確率質量を持つ
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
        lap_Q = Laplace(b=c/eps2).to_dist()
        noisy_Q = Add.apply(Q, lap_Q)

        # 閾値と比較
        over = GE(noisy_Q, T)

        # TRUEならノイズ付きクエリ値、FALSEなら-1000.0
        output_val = mux(over, noisy_Q, FALSE_SENTINEL)

        # 打ち切り後はNANを出力
        out_i = mux(broken, NAN, output_val)
        result.append(out_i)

        # カウンタを更新（打ち切り後は加算しない）
        inc = mux(broken, 0, over)
        count = Add.apply(count, inc)
        broken = GE(count, c)

    return result
