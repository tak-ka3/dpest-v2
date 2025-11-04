"""
MUX (multiplexer) 演算とセンチネル値の実装。

MUX は条件に基づいて値を選択する演算で、
センチネル値 (NAN, PAD) は特殊な状態を表現します。
"""

from typing import Union
from ..core import Dist, Node
from .condition_op import Condition


# センチネル値の定義
NAN = float('nan')  # 未定義/打ち切り後の値
PAD = -999999.0     # パディング値（未使用のスロット）


class MUX:
    """MUX (multiplexer) 演算: 条件に基づいて2つの値から1つを選択。

    MUX(condition, value_if_true, value_if_false) は以下と等価:
    - condition が真 (>= 0.5) のとき value_if_true を返す
    - condition が偽 (< 0.5) のとき value_if_false を返す
    """

    @staticmethod
    def apply(cond_dist: Dist, true_val: Union[Dist, float],
              false_val: Union[Dist, float]) -> Dist:
        """条件に基づいて値を選択。

        引数:
            cond_dist: 条件を表す {0,1} 上の分布
            true_val: 条件が真のときの値（分布または定数）
            false_val: 条件が偽のときの値（分布または定数）

        戻り値:
            選択された値の分布
        """
        # 定数を分布に変換
        if not isinstance(true_val, Dist):
            true_dist = Dist.deterministic(float(true_val))
        else:
            true_dist = true_val

        if not isinstance(false_val, Dist):
            false_dist = Dist.deterministic(float(false_val))
        else:
            false_dist = false_val

        # Condition演算を使って混合分布を作成
        result = Condition.apply(cond_dist, true_dist, false_dist)

        # ノード情報を更新
        deps = cond_dist.dependencies | true_dist.dependencies | false_dist.dependencies
        inputs = [
            getattr(cond_dist, 'node', None),
            getattr(true_dist, 'node', None),
            getattr(false_dist, 'node', None)
        ]
        inputs = [n for n in inputs if n is not None]
        node = Node(op='MUX', inputs=inputs, dependencies=set(deps))
        result.node = node

        return result


def mux(cond_dist: Dist, true_val: Union[Dist, float],
        false_val: Union[Dist, float]) -> Dist:
    """MUX.apply を呼び出すための簡易関数。"""
    return MUX.apply(cond_dist, true_val, false_val)
