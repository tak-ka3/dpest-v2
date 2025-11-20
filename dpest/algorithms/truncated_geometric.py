"""Truncated Geometric機構 (Algorithm 4.8)。

依存関係を自動検出し、サンプリングフォールバックに対応した実装。
branchオペレーションが依存関係を自動追跡し、@auto_dist()デコレータが
必要に応じてサンプリングモードに自動的に切り替えます。
"""

import math
from typing import List

from ..core import Dist
from ..operations import geq, branch
from ..noise import Uniform
from .registry import auto_dist


def _compute_f(c: int, eps: float, n: int) -> List[float]:
    """Truncated Geometric機構の累積分布関数Fを計算 (Alg. 4.8)。"""
    k = math.ceil(math.log(2.0 / eps))
    d = int((2 ** (k + 1) + 1) * ((2 ** k + 1) ** (n - 1)))

    f = [0] * (n + 1)

    for z in range(0, c):
        a = 2 ** (k * (c - z))
        b = (2 ** k + 1) ** (n - (c - z))
        f[z] = a * b

    for z in range(c, n):
        a = 2 ** (k * (z - c + 1))
        b = (2 ** k + 1) ** (n - 1 - (z - c))
        f[z] = d - a * b

    f[n] = d

    return f


def _input_scalar_to_array(c_dist: Dist, func, size: int) -> List[Dist]:
    """スカラー分布から配列分布を生成。"""
    if len(c_dist.atoms) == 1 and c_dist.atoms[0][1] == 1.0:
        c_val = int(c_dist.atoms[0][0])
        array = func(c_val)
        return [Dist.deterministic(float(array[i])) for i in range(size)]

    result_arrays = {}
    for c_val, c_prob in c_dist.atoms:
        c_int = int(c_val)
        array = func(c_int)
        for i in range(size):
            if i not in result_arrays:
                result_arrays[i] = {}
            val = float(array[i])
            if val not in result_arrays[i]:
                result_arrays[i][val] = 0.0
            result_arrays[i][val] += c_prob

    return [Dist.from_atoms(list(result_arrays[i].items())) for i in range(size)]


@auto_dist()
def truncated_geometric(c: List[Dist], eps: float = 0.1, n: int = 5) -> Dist:
    """Truncated Geometric機構を適用 (Algorithm 4.8)。

    この実装は依存関係を自動検出し、必要に応じてサンプリングモードに
    自動的に切り替わります。プログラマは依存関係を意識する必要はありません。

    動作の仕組み:
    1. branchオペレーションが確率変数間の共通依存性を検出
    2. 条件付き確率情報がない場合、needs_sampling=Trueを設定
    3. @auto_dist()デコレータが自動的にサンプリングモードに切り替え
    4. 正しい確率分布を計算

    Args:
        c: カウンティングクエリの結果分布（単一要素のリスト）
        eps: プライバシーパラメータ ε
        n: 個人数

    Returns:
        {0, ..., n} 上の確率質量関数
    """
    if len(c) != 1:
        raise ValueError(f"truncated_geometric expects a single value, got {len(c)} values")

    c_dist = c[0]
    k = math.ceil(math.log(2.0 / eps))
    d = int((2 ** (k + 1) + 1) * ((2 ** k + 1) ** (n - 1)))

    def compute_f_wrapper(c_val: int) -> List[float]:
        return _compute_f(c_val, eps, n)

    Arr = _input_scalar_to_array(c_dist, compute_f_wrapper, n + 1)
    u = Uniform(low=1, high=d).to_dist()
    z = Dist.deterministic(0.0)

    # branchが依存関係を自動検出し、@auto_dist()が自動的にサンプリングに切り替え
    for idx in reversed(range(n + 1)):
        z = branch(geq(Arr[idx], u), float(idx), z)

    return z
