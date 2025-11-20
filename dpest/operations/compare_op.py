"""
比較演算を扱うモジュール。

本モジュールは確率分布フレームワーク内で、
乱数同士の比較を処理するための演算を提供する。
"""

from typing import Union
import numpy as np

from ..core import Dist, Node
from .operations import Add, Affine


class Compare:
    """指示値の分布を返す比較演算。"""

    @staticmethod
    def geq(x_dist: Dist, y: Union[Dist, float]) -> Dist:
        """指示値 ``[X >= Y]`` の分布を返す。

        引数:
            x_dist: 変数 ``X`` の分布。
            y: 変数 ``Y`` の分布または定数。

        戻り値:
            事象 ``X >= Y`` に対応する {0,1} 上の分布。
        """
        # ``y`` が確率変数のときは ``X - Y`` と 0 を比較する。
        # 減算を基本演算 Add/Affine で表し計算グラフの情報を失わない。
        if isinstance(y, Dist):
            diff = Add.apply(x_dist, Affine.apply(y, -1.0, 0.0))
            return Compare.geq(diff, 0.0)

        threshold = float(y)
        prob = 0.0

        # 離散部分
        if x_dist.atoms:
            for val, weight in x_dist.atoms:
                if val >= threshold:
                    prob += weight

        # 連続部分: x >= threshold の領域で密度を積分する。
        # 任意の格子でも安定するよう ``trapz`` を利用する。
        if x_dist.density and 'x' in x_dist.density:
            x_grid = x_dist.density['x']
            f_grid = x_dist.density['f']
            mask = x_grid >= threshold
            if np.any(mask):
                prob += np.trapz(f_grid[mask], x_grid[mask])

        prob = min(max(prob, 0.0), 1.0)
        # 下流の解析で依存関係が分かるよう ``x_dist``
        # （必要なら ``y``）への依存を記録する。
        deps = set(x_dist.dependencies)
        inputs = [getattr(x_dist, 'node', None)]
        if isinstance(y, Dist):
            deps |= y.dependencies
            inputs.append(getattr(y, 'node', None))
        inputs = [n for n in inputs if n is not None]
        node = Node(op='CompareGEQ', inputs=inputs, dependencies=set(deps))
        result = Dist.from_atoms([(1.0, prob), (0.0, 1.0 - prob)],
                                 dependencies=set(deps), node=node)
        result._sample_func = (
            (lambda cache, xd=x_dist, y_val=y: 1.0 if xd._sample(cache) >= (y_val._sample(cache) if isinstance(y_val, Dist) else float(y_val)) else 0.0)
            if isinstance(y, Dist)
            else (lambda cache, xd=x_dist, y_const=float(y): 1.0 if xd._sample(cache) >= y_const else 0.0)
        )
        return result


def geq(x_dist: Dist, y: Union[Dist, float]) -> Dist:
    """``Compare.geq`` を呼び出すための簡易関数。"""
    return Compare.geq(x_dist, y)
