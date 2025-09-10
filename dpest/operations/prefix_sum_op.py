"""Prefix sum operation for distributions."""

from typing import List
from ..core import Dist
from .operations import Add


class PrefixSum:
    """Prefix sum: cumulative addition of independent distributions."""

    @staticmethod
    def apply(distributions: List[Dist]) -> List[Dist]:
        """
        連続する分布のリストに対して累積和を計算する。
        各分布は互いに独立であると仮定する。

        Args:
            distributions: 加算する分布のリスト

        Returns:
            各ステップの累積分布のリスト
        """
        results: List[Dist] = []
        cumulative = Dist.deterministic(0.0)
        for dist in distributions:
            cumulative = Add.apply(cumulative, dist)
            results.append(cumulative)
        return results


def prefix_sum_distributions(distributions: List[Dist]) -> List[Dist]:
    """便利関数: PrefixSum.apply を呼び出す"""
    return PrefixSum.apply(distributions)
