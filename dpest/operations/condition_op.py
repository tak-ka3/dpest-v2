"""
条件分岐と比較を扱う演算群。

本モジュールは確率分布フレームワーク内で、条件分岐や
乱数同士の比較を処理するための演算を提供する。
"""

from typing import Union
import numpy as np
from scipy import interpolate

from ..core import Dist, merge_atoms, Node
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


class Condition:
    """条件分岐による混合演算。

    {0,1} 上の条件分布と 2 つの枝の分布から全体の混合分布を求める。
    """

    @staticmethod
    def apply(cond_dist: Dist, true_dist: Dist, false_dist: Dist) -> Dist:
        """ ``P(E)*true + P(¬E)*false`` の混合分布を返す。

        引数:
            cond_dist: 事象 ``E`` を表す {0,1} 上の分布。
            true_dist: ``E`` が真のときに用いる分布。
            false_dist: ``E`` が偽のときに用いる分布。
        """
        p_true = 0.0
        for val, weight in cond_dist.atoms:
            if val >= 0.5:
                p_true += weight
        p_true = min(max(p_true, 0.0), 1.0)
        p_false = 1.0 - p_true

        result_atoms = []
        if true_dist.atoms:
            result_atoms.extend((v, w * p_true) for v, w in true_dist.atoms)
        if false_dist.atoms:
            result_atoms.extend((v, w * p_false) for v, w in false_dist.atoms)

        result_density = {}
        if true_dist.density or false_dist.density:
            # 枝のグリッドと密度を取り出す。欠けている要素は空配列とし、
            # その後の処理が一様に扱えるようにする。
            x_true = true_dist.density.get('x', np.array([])) if true_dist.density else np.array([])
            f_true = true_dist.density.get('f', np.array([])) if true_dist.density else np.array([])
            dx_true = true_dist.density.get('dx') if true_dist.density else None
            x_false = false_dist.density.get('x', np.array([])) if false_dist.density else np.array([])
            f_false = false_dist.density.get('f', np.array([])) if false_dist.density else np.array([])
            dx_false = false_dist.density.get('dx') if false_dist.density else None

            if x_true.size > 0 and x_false.size > 0:
                # 両方の枝に密度がある場合は共通グリッドに再標本化し、
                # 確率を整列させてから重み付けして混合する。
                dx = min(dx_true, dx_false)
                min_x = min(x_true[0], x_false[0])
                max_x = max(x_true[-1], x_false[-1])
                n_points = int((max_x - min_x) / dx) + 1
                x_grid = np.linspace(min_x, max_x, n_points)
                f_true_interp = interpolate.interp1d(x_true, f_true, bounds_error=False, fill_value=0.0)
                f_false_interp = interpolate.interp1d(x_false, f_false, bounds_error=False, fill_value=0.0)
                f_mix = p_true * f_true_interp(x_grid) + p_false * f_false_interp(x_grid)
                result_density = {'x': x_grid, 'f': f_mix, 'dx': dx}
            elif x_true.size > 0:
                # 真の枝のみ密度を持つ場合はそのままスケールする。
                result_density = {'x': x_true, 'f': p_true * f_true, 'dx': dx_true}
            elif x_false.size > 0:
                # 偽の枝のみ密度を持つ場合はそのままスケールする。
                result_density = {'x': x_false, 'f': p_false * f_false, 'dx': dx_false}

        result_atoms = merge_atoms(result_atoms)
        # 出力ノードは条件・真の枝・偽の枝のすべてに依存する。
        # 計算グラフのリンクを保持し、上位解析で値の合成過程を追跡できるようにする。
        deps = cond_dist.dependencies | true_dist.dependencies | false_dist.dependencies
        inputs = [getattr(cond_dist, 'node', None),
                  getattr(true_dist, 'node', None),
                  getattr(false_dist, 'node', None)]
        inputs = [n for n in inputs if n is not None]
        node = Node(op='Condition', inputs=inputs, dependencies=set(deps))
        result = Dist(atoms=result_atoms,
                      density=result_density if result_density else None,
                      dependencies=set(deps),
                      node=node)
        result._sample_func = lambda cache, cond=cond_dist, t_dist=true_dist, f_dist=false_dist: (
            t_dist._sample(cache) if cond._sample(cache) >= 0.5 else f_dist._sample(cache)
        )
        result.normalize()
        return result


def geq(x_dist: Dist, y: Union[Dist, float]) -> Dist:
    """ ``Compare.geq`` を呼び出すための簡易関数。"""
    return Compare.geq(x_dist, y)


def condition_mixture(cond_dist: Dist, true_dist: Dist, false_dist: Dist) -> Dist:
    """ ``Condition.apply`` を呼び出すための簡易関数。"""
    return Condition.apply(cond_dist, true_dist, false_dist)
