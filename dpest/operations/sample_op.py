"""
サンプルベースの機構分布近似

任意の機構から得られるサンプルを用いて確率分布を構築します。
これにより、SparseVectorTechnique など複雑なアルゴリズムにも
演算フレームワークを適用できます。
"""

from typing import Callable, Union, List
import numpy as np
from ..core import Dist, Interval


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
        return Sampled.from_samples(samples, bins)

    @staticmethod
    def from_samples(samples: np.ndarray, bins: int = 100) -> Union[Dist, List[Dist]]:
        """既存のサンプル配列から分布を構築

        Args:
            samples: サンプル配列 (n_samples,) または (n_samples, k)
            bins: 連続値の場合のヒストグラム分割数。

        Returns:
            Dist または Dist のリスト。
        """
        samples = np.asarray(samples)
        if samples.ndim == 1:
            dist = Sampled._samples_to_dist(samples, bins)
            row_key = ('sampled_row', id(samples))

            def sampler(cache, arr=samples):
                if arr.size == 0:
                    return float('nan')
                row = cache.setdefault(row_key, np.random.randint(arr.shape[0]))
                return float(arr[row])

            dist._sample_func = sampler
            return dist

        row_key = ('sampled_row', id(samples))
        dists = []
        for i in range(samples.shape[1]):
            dist = Sampled._samples_to_dist(samples[:, i], bins)

            def sampler(cache, idx=i, arr=samples):
                if arr.shape[0] == 0:
                    return float('nan')
                row = cache.setdefault(row_key, np.random.randint(arr.shape[0]))
                return float(arr[row, idx])

            dist._sample_func = sampler
            dists.append(dist)

        for dist in dists:
            dist._joint_samples = samples

        return dists

    @staticmethod
    def _samples_to_dist(samples: np.ndarray, bins: int) -> Dist:
        """1次元サンプルからDistを構築"""
        samples = samples.astype(float)

        # NaNを含むかチェック
        has_nan = np.any(np.isnan(samples))

        if has_nan:
            # NaNを分離して処理
            nan_mask = np.isnan(samples)
            non_nan_samples = samples[~nan_mask]
            nan_count = np.sum(nan_mask)
            total = samples.size
            nan_prob = nan_count / total if total > 0 else 0.0

            if len(non_nan_samples) == 0:
                # 全てNaNの場合
                return Dist.from_atoms([(float('nan'), 1.0)])

            # 非NaN値の分布を構築
            unique_vals = np.unique(non_nan_samples)

            # 離散的な値が少ない場合は点質量として扱う
            if np.all(np.mod(unique_vals, 1) == 0) and len(unique_vals) <= bins // 2:
                counts = [(v, np.sum(non_nan_samples == v)) for v in unique_vals]
                # NaNと非NaN値の両方を含むatoms
                atoms = [(float(v), c / total) for v, c in counts]
                atoms.append((float('nan'), nan_prob))
                return Dist.from_atoms(atoms)
            else:
                # 連続値の場合 - NaNは別途処理が必要
                # 今のところ離散化して対応
                hist, edges = np.histogram(non_nan_samples, bins=bins, density=False)
                atoms = []
                for i in range(len(hist)):
                    if hist[i] > 0:
                        center = (edges[i] + edges[i+1]) / 2
                        atoms.append((center, hist[i] / total))
                atoms.append((float('nan'), nan_prob))
                return Dist.from_atoms(atoms)
        else:
            # NaNを含まない場合（既存のロジック）
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
