"""
NumericalSVT - 分布ベース実装

Numerical Sparse Vector Technique from:
    Y. Wang, Z. Ding, G. Wang, D. Kifer, D. Zhang. 2019.
    Proving differential privacy with shadow execution.
    PLDI 2019.

NumericalSVTの特徴: TRUEの時にノイズ付きクエリ値を出力
"""

from typing import List
from ..core import Dist
from ..noise import Laplace
from ..operations import add, affine, branch, geq
from .registry import auto_dist

# センチネル値
NAN = float('nan')


@auto_dist()
def numerical_svt(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]:
    """
    NumericalSVTアルゴリズムの分布ベース実装

    Args:
        queries: クエリ結果の分布のリスト
        eps: プライバシーパラメータ（デフォルト: 0.1）
        t: 閾値（デフォルト: 1.0）
        c: 最大出力回数（cutoff、デフォルト: 2）

    Returns:
        各クエリに対する出力分布のリスト
        各分布は {ノイズ付き値: TRUE, 0.0: FALSE, NAN: ABORTED} の確率質量を持つ
    """
    # 閾値にノイズを追加: T = t + Laplace(b=3/eps)
    lap_rho1 = Laplace(b=3/eps).to_dist()
    T = affine(lap_rho1, 1.0, t)

    # カウンタと打ち切りフラグ
    count = Dist.deterministic(0.0)
    broken = Dist.deterministic(0.0)
    result = []

    for Q in queries:
        # 比較用のノイズ: rho2
        lap_rho2 = Laplace(b=6*c/eps).to_dist()
        noisy_Q_cmp = add(Q, lap_rho2)

        # 閾値と比較
        over = geq(noisy_Q_cmp, T)

        # 出力用のノイズ: rho3
        lap_rho3 = Laplace(b=3*c/eps).to_dist()
        noisy_Q_out = add(Q, lap_rho3)

        # TRUEならノイズ付きクエリ値、FALSEなら0.0
        output_val = branch(over, noisy_Q_out, 0.0)

        # 打ち切り後はNANを出力
        out_i = branch(broken, NAN, output_val)
        result.append(out_i)

        # カウンタを更新（打ち切り後は加算しない）
        inc = branch(broken, 0, over)
        count = add(count, inc)
        broken = geq(count, c)

    return result
