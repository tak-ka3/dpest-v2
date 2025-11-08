"""Sum of noisy maxima from two vectors."""

from typing import List

from ..core import Dist
from ..operations import vector_add, vector_max, add
from ..noise import create_laplace_noise
from .registry import auto_dist


@auto_dist()
def noisy_max_sum(
    values: List[Dist],
    eps: float = 0.1,
    split_index: int | None = None,
) -> Dist:
    """
    Add Laplace noise to two subvectors, take their maxima, and sum.

    Args:
        values: ベクトル1とベクトル2を連結した Dist リスト。
        eps: 各要素に与えるラプラスノイズの ε (scale=1/eps)。
        split_index: ベクトル1の長さ（None の場合は半分で分割）。

    Returns:
        2つの noisy max の和を表す Dist。
    """
    if split_index is None:
        if len(values) % 2 != 0:
            raise ValueError("noisy_max_sum requires even-length input when split_index is None")
        split_index = len(values) // 2

    vec1 = values[:split_index]
    vec2 = values[split_index:]
    if not vec1 or not vec2:
        raise ValueError("noisy_max_sum expects two non-empty vectors")

    def noisy_max(vec: List[Dist]) -> Dist:
        noise = create_laplace_noise(b=1 / eps, size=len(vec))
        noisy_vec = vector_add(vec, noise)
        return vector_max(noisy_vec)

    max1 = noisy_max(vec1)
    max2 = noisy_max(vec2)
    return add(max1, max2)
