"""Truncated Geometric mechanism."""

import math
from typing import List

from ..core import Dist
from ..operations import geq, mux
from .registry import auto_dist


def _compute_f(c: int, eps: float, n: int) -> List[float]:
    """Compute the F function as defined in Step 2 of Alg. 4.8.

    Implements the cumulative distribution function for the Truncated Geometric mechanism.

    Args:
        c: カウンティングクエリの結果 (0 <= c <= n)
        eps: プライバシーパラメータ ε
        n: カウンティングクエリの対象となる個人数

    Returns:
        List f of shape (n+1,), where f[z] = F(z) for z = 0, 1, ..., n

    Note:
        この関数はAlg. 4.8の累積分布関数を計算します。
        参考実装（dpest_reference）と同じロジックです。
    """
    k = math.ceil(math.log(2.0 / eps))
    d = int((2 ** (k + 1) + 1) * ((2 ** k + 1) ** (n - 1)))

    f = [0] * (n + 1)

    # For interval [0, c)
    for z in range(0, c):
        a = 2 ** (k * (c - z))
        b = (2 ** k + 1) ** (n - (c - z))
        f[z] = a * b

    # For interval [c, n)
    for z in range(c, n):
        a = 2 ** (k * (z - c + 1))
        b = (2 ** k + 1) ** (n - 1 - (z - c))
        f[z] = d - a * b

    # For n
    f[n] = d

    return f


def _input_scalar_to_array(c_dist: Dist, func, size: int) -> List[Dist]:
    """InputScalarToArray: 入力スカラーから配列を生成

    参考実装の InputScalarToArray(size=n+1, func=compute_f) に対応。
    c_dist の各値に対して func を適用し、配列を生成します。

    Args:
        c_dist: 入力スカラーの分布
        func: c を受け取り、配列を返す関数
        size: 配列のサイズ

    Returns:
        List[Dist]: 各インデックスに対応する分布のリスト
    """
    # c_dist が確定値の場合
    if len(c_dist.atoms) == 1 and c_dist.atoms[0][1] == 1.0:
        c_val = int(c_dist.atoms[0][0])
        array = func(c_val)
        return [Dist.deterministic(float(array[i])) for i in range(size)]

    # c_dist が分布の場合、各cに対して配列を計算して混合
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

    # 各インデックスの分布を構築
    return [Dist.from_atoms(list(result_arrays[i].items())) for i in range(size)]


@auto_dist()
def truncated_geometric(c: List[Dist], eps: float = 0.1, n: int = 5) -> Dist:
    """Apply Truncated Geometric mechanism to a counting query result.

    Args:
        c: カウンティングクエリの結果の分布（単一要素のリスト）
        eps: プライバシーパラメータ ε
        n: カウンティングクエリの対象となる個人数

    Returns:
        Dist: {0, ..., n} 上の確率質量関数

    Note:
        Thm 4.7より、GeoSample(eps, n)は ln(1 + 2^-ceil(ln(2/eps)))-DP を満たす

    Implementation:
        参考実装（dpest_reference/test/alg/truncated_geometric.py）と完全に同じ:

        ```python
        Arr = InputScalarToArray(size=n+1, func=compute_f)
        u = Uni(1, d+1)
        z = 0

        for idx in reversed(range(n+1)):
            z = Br(u, Arr[idx], z, idx)
        Y = z
        ```

        これを既存オペレーション（geq, mux）で実装します。
    """
    # c は List[Dist] だが、単一要素のはず
    if len(c) != 1:
        raise ValueError(f"truncated_geometric expects a single value, got {len(c)} values")

    c_dist = c[0]

    # パラメータ計算
    k = math.ceil(math.log(2.0 / eps))
    d = int((2 ** (k + 1) + 1) * ((2 ** k + 1) ** (n - 1)))

    # Arr = InputScalarToArray(size=n+1, func=compute_f)
    # c に応じて compute_f を実行し、F配列を生成
    def compute_f_wrapper(c_val: int) -> List[float]:
        return _compute_f(c_val, eps, n)

    Arr = _input_scalar_to_array(c_dist, compute_f_wrapper, n + 1)

    # u = Uni(1, d+1)
    # 離散一様分布: 各整数値 1, 2, ..., d が等確率
    u_atoms = [(float(i), 1.0 / d) for i in range(1, d + 1)]
    u = Dist.from_atoms(u_atoms)

    # z = 0
    z = Dist.deterministic(0.0)

    # for idx in reversed(range(n+1)):
    #     z = Br(u, Arr[idx], z, idx)
    #
    # Br(input1, input2, output1, output2) は input1 >= input2 なら output1、そうでなければ output2
    # 参考実装の機構（line 46）: z[f[idx] >= u] = idx
    # つまり f[idx] >= u なら z = idx
    # よって Br(Arr[idx], u, idx, z) と解釈:
    #   - Arr[idx] >= u なら idx
    #   - Arr[idx] < u なら z
    for idx in reversed(range(n + 1)):
        # Arr[idx] >= u という条件
        condition = geq(Arr[idx], u)
        # true なら idx、false なら z
        z = mux(condition, float(idx), z)

    # Y = z
    return z
