"""
ベクトル演算の実装

確率分布のリストに対するベクトル演算を提供します。
"""

from typing import List, Union
from ..core import Dist
from .operations import Add
from .max_op import max_op, min_op
from .argmax_op import argmax


def vector_add(x_list: List[Dist], y_list: Union[List[Dist], Dist]) -> List[Dist]:
    """ベクトル（またはスカラー）加算を行う

    Args:
        x_list: 分布のリスト
        y_list: 分布のリストまたは単一の分布（ブロードキャスト）

    Returns:
        要素ごとの加算結果のリスト

    Examples:
        >>> # ベクトル + ベクトル
        >>> result = vector_add([d1, d2, d3], [n1, n2, n3])
        >>> # ベクトル + スカラー
        >>> result = vector_add([d1, d2, d3], noise)
    """
    if isinstance(y_list, Dist):
        return [Add.apply(x, y_list) for x in x_list]
    if len(x_list) != len(y_list):
        raise ValueError("Vector lengths must match")
    return [Add.apply(x, y) for x, y in zip(x_list, y_list)]


def vector_argmax(distributions: List[Dist]) -> Dist:
    """ベクトルのargmaxを計算

    Args:
        distributions: 分布のリスト

    Returns:
        最大値を取るインデックスの分布（離散分布）

    Examples:
        >>> noisy_values = vector_add(values, noises)
        >>> winner_idx = vector_argmax(noisy_values)
    """
    return argmax(distributions)


def vector_max(distributions: List[Dist]) -> Dist:
    """ベクトルのmaxを計算

    Args:
        distributions: 分布のリスト

    Returns:
        最大値の分布

    Examples:
        >>> noisy_values = vector_add(values, noises)
        >>> max_value = vector_max(noisy_values)
    """
    return max_op(distributions)


def vector_min(distributions: List[Dist]) -> Dist:
    """ベクトルのminを計算

    Args:
        distributions: 分布のリスト

    Returns:
        最小値の分布

    Examples:
        >>> noisy_values = vector_add(values, noises)
        >>> min_value = vector_min(noisy_values)
    """
    return min_op(distributions)
