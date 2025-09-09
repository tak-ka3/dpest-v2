"""
サンプルベースの機構分布近似

任意の機構から得られるサンプルを用いて確率分布を構築します。
これにより、SparseVectorTechnique など複雑なアルゴリズムにも
演算フレームワークを適用できます。
"""

from typing import Callable, Union, List
import numpy as np
from core import Dist, Interval


class Sampled:
    """サンプルから経験的分布を構築する操作"""

    @staticmethod
    def apply(sample_fn: Callable[[int], np.ndarray], n_samples: int = 1000,
              bins: int = 100) -> Union[Dist, List[Dist]]:
        """サンプル生成関数から分布を近似

        Args:
            sample_fn: `n_samples` を引数に取り `(n_samples,)` もしくは
                `(n_samples, k)` 形状のサンプルを返す関数。
            n_samples: 使用するサンプル数。
            bins: 連続値の場合のヒストグラム分割数。

        Returns:
            Dist または Dist のリスト。
        """
        samples = np.asarray(sample_fn(n_samples))

        if samples.ndim == 1:
            return Sampled._samples_to_dist(samples, bins)
        else:
            return [Sampled._samples_to_dist(samples[:, i], bins)
                    for i in range(samples.shape[1])]

    @staticmethod
    def _samples_to_dist(samples: np.ndarray, bins: int) -> Dist:
        """1次元サンプルからDistを構築"""
        samples = samples.astype(float)
        unique_vals = np.unique(samples)

        # 離散的な値が少ない場合は点質量として扱う
        if np.all(np.mod(unique_vals, 1) == 0) and len(unique_vals) <= bins // 2:
            counts = [(v, np.sum(samples == v)) for v in unique_vals]
            total = samples.size
            atoms = [(float(v), c / total) for v, c in counts]
            return Dist.from_atoms(atoms)

        # 連続値の場合はヒストグラムで近似
        hist, edges = np.histogram(samples, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        dx = edges[1] - edges[0]
        dist = Dist(density={'x': centers, 'f': hist, 'dx': dx},
                    support=[Interval(centers[0], centers[-1])])
        dist.normalize()
        return dist


def sampled_distribution(sample_fn: Callable[[int], np.ndarray], n_samples: int = 1000,
                         bins: int = 100) -> Union[Dist, List[Dist]]:
    """便利関数: Sampled.apply のラッパー"""
    return Sampled.apply(sample_fn, n_samples=n_samples, bins=bins)
