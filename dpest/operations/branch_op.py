"""
依存関係を自動追跡するbranch演算の実装。

このモジュールは、同じ確率変数を複数回参照する場合に
条件付き確率を自動計算するbranch演算を提供します。
"""

from typing import Union, Dict, Tuple
import numpy as np
from scipy import interpolate

from ..core import Dist, Node, merge_atoms
from .compare_op import geq

# センチネル値の定義
NAN = float('nan')  # 未定義/打ち切り後の値
PAD = -999999.0     # パディング値（未使用のスロット）


class Condition:
    """条件分岐による混合演算。

    {0,1} 上の条件分布と 2 つの枝の分布から全体の混合分布を求める。
    内部実装クラスであり、ユーザーはbranch()関数を使用すべき。
    """

    @staticmethod
    def apply(cond_dist: Dist, true_dist: Dist, false_dist: Dist) -> Dist:
        """ ``P(E)*true + P(¬E)*false`` の混合分布を返す。

        引数:
            cond_dist: 事象 ``E`` を表す {0,1} 上の分布。
            true_dist: ``E`` が真のときに用いる分布。
            false_dist: ``E`` が偽のときに用いる分布。

        注意:
            cond_dist と false_dist が共通の依存性を持つ場合（同じ確率変数に依存）、
            条件付き確率を用いて正しく計算する。
        """
        p_true = 0.0
        for val, weight in cond_dist.atoms:
            if val >= 0.5:
                p_true += weight
        p_true = min(max(p_true, 0.0), 1.0)
        p_false = 1.0 - p_true

        # 共通の依存性を検出
        shared_deps = cond_dist.dependencies & false_dist.dependencies

        result_atoms = []

        # 共通依存性がある場合、条件付き確率で計算を試みる
        has_conditional_info = False
        if shared_deps and false_dist.atoms:
            # false_dist が _condition_given_false 属性を持つ場合、それを使用
            # これは「cond=False の下での条件付き分布」を表す
            if hasattr(false_dist, '_condition_given_false'):
                has_conditional_info = True
                if true_dist.atoms:
                    result_atoms.extend((v, w * p_true) for v, w in true_dist.atoms)
                # 条件付き確率を使用
                for v, w_cond in false_dist._condition_given_false:
                    result_atoms.append((v, w_cond * p_false))
            else:
                # フォールバック: 通常の独立混合（不正確だがneeds_samplingで後で修正）
                if true_dist.atoms:
                    result_atoms.extend((v, w * p_true) for v, w in true_dist.atoms)
                if false_dist.atoms:
                    result_atoms.extend((v, w * p_false) for v, w in false_dist.atoms)
        else:
            # 通常の独立混合
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

        # 共通依存性があるが条件付き確率情報がない場合、サンプリングが必要
        needs_sampling = bool(shared_deps) and not has_conditional_info
        node = Node(op='Condition', inputs=inputs, dependencies=set(deps),
                   needs_sampling=needs_sampling)
        result = Dist(atoms=result_atoms,
                      density=result_density if result_density else None,
                      dependencies=set(deps),
                      node=node)
        result._sample_func = lambda cache, cond=cond_dist, t_dist=true_dist, f_dist=false_dist: (
            t_dist._sample(cache) if cond._sample(cache) >= 0.5 else f_dist._sample(cache)
        )
        result.normalize()
        return result


class Branch:
    """依存関係を考慮した条件分岐演算。

    Br(condition, value_if_true, value_if_false) は、
    condition と value_if_false が共通の確率変数に依存する場合、
    条件付き確率を自動的に計算します。
    """

    @staticmethod
    def apply(cond_dist: Dist, true_val: Union[Dist, float],
              false_val: Union[Dist, float]) -> Dist:
        """条件に基づいて値を選択（依存関係を自動追跡）。

        Args:
            cond_dist: 条件を表す {0,1} 上の分布
            true_val: 条件が真のときの値（分布または定数）
            false_val: 条件が偽のときの値（分布または定数）

        Returns:
            選択された値の分布（条件付き確率を考慮）

        Note:
            cond_dist と false_val に共通の依存性がある場合、
            自動的に条件付き確率を計算します。
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

        # 共通の依存性を検出
        shared_deps = cond_dist.dependencies & false_dist.dependencies

        # 条件の確率を計算
        p_true = 0.0
        for val, weight in cond_dist.atoms:
            if val >= 0.5:
                p_true += weight
        p_true = min(max(p_true, 0.0), 1.0)
        p_false = 1.0 - p_true

        # 共通依存性がある場合、条件付き確率を自動計算
        if shared_deps:
            # 条件付き確率情報があるかチェック
            has_conditional_info = (hasattr(false_dist, '_u_ranges') and
                                   hasattr(false_dist, '_condition_given_false'))

            if has_conditional_info:
                # uの範囲情報を使って条件付き確率を計算
                result_atoms = []

                # true_distの部分（条件が真の場合）
                if true_dist.atoms:
                    for v, w in true_dist.atoms:
                        result_atoms.append((v, w * p_true))

                # false_distの条件付き確率（条件が偽の場合）
                for v, w_cond in false_dist._condition_given_false:
                    result_atoms.append((v, w_cond * p_false))
            else:
                # 条件付き確率情報がない場合、サンプリングにフォールバック
                result_atoms = []
                if true_dist.atoms:
                    for v, w in true_dist.atoms:
                        result_atoms.append((v, w * p_true))
                if false_dist.atoms:
                    for v, w in false_dist.atoms:
                        result_atoms.append((v, w * p_false))

            # アトムをマージ
            from ..core import merge_atoms
            result_atoms = merge_atoms(result_atoms)

            # 密度の処理（Condition.apply()と同じロジック）
            import numpy as np
            from scipy import interpolate
            result_density = {}
            if true_dist.density or false_dist.density:
                x_true = true_dist.density.get('x', np.array([])) if true_dist.density else np.array([])
                f_true = true_dist.density.get('f', np.array([])) if true_dist.density else np.array([])
                dx_true = true_dist.density.get('dx') if true_dist.density else None
                x_false = false_dist.density.get('x', np.array([])) if false_dist.density else np.array([])
                f_false = false_dist.density.get('f', np.array([])) if false_dist.density else np.array([])
                dx_false = false_dist.density.get('dx') if false_dist.density else None

                if x_true.size > 0 and x_false.size > 0:
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
                    result_density = {'x': x_true, 'f': p_true * f_true, 'dx': dx_true}
                elif x_false.size > 0:
                    result_density = {'x': x_false, 'f': p_false * f_false, 'dx': dx_false}

            # 依存関係を結合
            deps = cond_dist.dependencies | true_dist.dependencies | false_dist.dependencies
            inputs = [
                getattr(cond_dist, 'node', None),
                getattr(true_dist, 'node', None),
                getattr(false_dist, 'node', None)
            ]
            inputs = [n for n in inputs if n is not None]

            # 条件付き確率情報がない場合、サンプリングが必要
            needs_sampling = not has_conditional_info
            node = Node(op='Branch', inputs=inputs, dependencies=set(deps),
                       needs_sampling=needs_sampling)

            result = Dist(
                atoms=result_atoms,
                density=result_density if result_density else None,
                dependencies=set(deps),
                node=node
            )

            # サンプリング関数を設定
            result._sample_func = lambda cache, cond=cond_dist, t_dist=true_dist, f_dist=false_dist: (
                t_dist._sample(cache) if cond._sample(cache) >= 0.5 else f_dist._sample(cache)
            )

            result.normalize()
            return result

        else:
            # 共通依存性がない場合、通常のCondition演算を使用
            result = Condition.apply(cond_dist, true_dist, false_dist)

            # ノード情報を更新
            inputs = [
                getattr(cond_dist, 'node', None),
                getattr(true_dist, 'node', None),
                getattr(false_dist, 'node', None)
            ]
            inputs = [n for n in inputs if n is not None]
            deps = cond_dist.dependencies | true_dist.dependencies | false_dist.dependencies
            node = Node(op='Branch', inputs=inputs, dependencies=set(deps))
            result.node = node

            return result


def branch(cond_dist: Dist, true_val: Union[Dist, float],
           false_val: Union[Dist, float]) -> Dist:
    """Branch.apply を呼び出すための簡易関数。

    Args:
        cond_dist: 条件を表す {0,1} 上の分布
        true_val: 条件が真のときの値（分布または定数）
        false_val: 条件が偽のときの値（分布または定数）

    Returns:
        選択された値の分布（依存関係を自動追跡）

    Examples:
        >>> u = Uniform(low=1, high=10).to_dist()
        >>> z = Dist.deterministic(0.0)
        >>> for idx in range(5):
        >>>     condition = geq(threshold[idx], u)
        >>>     z = branch(condition, idx, z)  # 依存関係を自動追跡
    """
    return Branch.apply(cond_dist, true_val, false_val)
