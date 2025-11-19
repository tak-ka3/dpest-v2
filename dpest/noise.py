"""
ノイズ機構の実装

差分プライバシーで使用されるノイズ分布を実装します。
"""

import numpy as np
from typing import Union, List
from .core import Dist, Interval, Node


class Laplace:
    """ラプラス分布ノイズ機構
    
    PDF: f(x) = (1/(2b)) * exp(-|x-μ|/b)
    """
    
    def __init__(self, b: float = 1.0, mu: float = 0.0, size: Union[int, None] = None):
        """
        Args:
            b: スケールパラメータ（分散 = 2b²）
            mu: 位置パラメータ（平均）
            size: ベクトルサイズ（Noneの場合はスカラー）
        """
        if b <= 0:
            raise ValueError(f"Scale parameter b must be positive, got {b}")
        
        self.b = b
        self.mu = mu
        self.size = size
    
    def to_dist(self, grid_size: int = 1000, support_range: float = None) -> Union[Dist, List[Dist]]:
        """ラプラス分布をDist形式に変換
        
        Args:
            grid_size: 格子点数
            support_range: サポート範囲（Noneの場合は自動設定）
            
        Returns:
            Dist: スカラーの場合
            List[Dist]: ベクトルの場合
        """
        if support_range is None:
            # 99.9%の確率質量を含む範囲を設定
            support_range = self.b * 7  #約99.9%
        
        x = np.linspace(self.mu - support_range, self.mu + support_range, grid_size)
        f = self._pdf(x)

        if self.size is None:
            def sampler(n, mu=self.mu, b=self.b):
                return np.random.laplace(mu, b, (n, 1))

            dist = Dist.from_density(
                x,
                f,
                sampler=sampler,
                sampler_index=0,
                sample_func=lambda cache, mu=self.mu, b=self.b: float(np.random.laplace(mu, b)),
            )
            node = Node(op='Laplace', inputs=[], dependencies=set(dist.dependencies))
            dist.node = node
            dist.support = [Interval(self.mu - support_range, self.mu + support_range)]
            return dist
        else:
            dists = []
            for _ in range(self.size):
                def sampler(n, mu=self.mu, b=self.b):
                    return np.random.laplace(mu, b, (n, 1))

                dist = Dist.from_density(
                    x,
                    f,
                    sampler=sampler,
                    sampler_index=0,
                    sample_func=lambda cache, mu=self.mu, b=self.b: float(np.random.laplace(mu, b)),
                )
                node = Node(op='Laplace', inputs=[], dependencies=set(dist.dependencies))
                dist.node = node
                dist.support = [Interval(self.mu - support_range, self.mu + support_range)]
                dists.append(dist)
            # 独立なラプラス分布のリストを返す
            return dists
    
    def _pdf(self, x: np.ndarray) -> np.ndarray:
        """確率密度関数"""
        return (1 / (2 * self.b)) * np.exp(-np.abs(x - self.mu) / self.b)
    
    def sample(self, n: int = 1) -> np.ndarray:
        """サンプリング（テスト用）"""
        if self.size is None:
            return np.random.laplace(self.mu, self.b, n)
        else:
            return np.random.laplace(self.mu, self.b, (n, self.size))
    
    def __repr__(self):
        size_str = f", size={self.size}" if self.size is not None else ""
        return f"Laplace(b={self.b}, mu={self.mu}{size_str})"


def create_laplace_noise(
    b: float,
    size: int = None,
    *,
    grid_size: int = 1000,
    support_range: float = None,
) -> Union[Dist, List[Dist]]:
    """便利関数：ラプラスノイズ分布を作成

    grid_size や support_range を指定することで、連続分布の離散化精度を
    制御できるようにする。精度を高めたい場合は grid_size を大きく設定する。
    """
    laplace = Laplace(b=b, size=size)
    return laplace.to_dist(grid_size=grid_size, support_range=support_range)


class Exponential:
    """指数分布ノイズ機構

    PDF: f(x) = (1/b) * exp(-(x-μ)/b) for x ≥ μ
    """

    def __init__(self, b: float = 1.0, mu: float = 0.0, size: Union[int, None] = None):
        """
        Args:
            b: スケールパラメータ（平均）
            mu: シフトパラメータ
            size: ベクトルサイズ（Noneの場合はスカラー）
        """
        if b <= 0:
            raise ValueError(f"Scale parameter b must be positive, got {b}")

        self.b = b
        self.mu = mu
        self.size = size

    def to_dist(self, grid_size: int = 1000, support_range: float = None) -> Union[Dist, List[Dist]]:
        """指数分布をDist形式に変換"""
        if support_range is None:
            # 十分な確率質量を含む範囲を設定
            support_range = self.b * 20

        x = np.linspace(self.mu, self.mu + support_range, grid_size)
        f = self._pdf(x)
        dx = x[1] - x[0]
        f = f / (np.sum(f) * dx)

        if self.size is None:
            def sampler(n, mu=self.mu, b=self.b):
                return np.random.exponential(scale=b, size=(n, 1)) + mu

            dist = Dist.from_density(
                x,
                f,
                sampler=sampler,
                sampler_index=0,
                sample_func=lambda cache, mu=self.mu, b=self.b: float(np.random.exponential(scale=b) + mu),
            )
            node = Node(op='Exponential', inputs=[], dependencies=set(dist.dependencies))
            dist.node = node
            dist.support = [Interval(self.mu, self.mu + support_range)]
            return dist
        else:
            dists = []
            for _ in range(self.size):
                def sampler(n, mu=self.mu, b=self.b):
                    return np.random.exponential(scale=b, size=(n, 1)) + mu

                dist = Dist.from_density(
                    x,
                    f,
                    sampler=sampler,
                    sampler_index=0,
                    sample_func=lambda cache, mu=self.mu, b=self.b: float(np.random.exponential(scale=b) + mu),
                )
                node = Node(op='Exponential', inputs=[], dependencies=set(dist.dependencies))
                dist.node = node
                dist.support = [Interval(self.mu, self.mu + support_range)]
                dists.append(dist)
            return dists

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        """確率密度関数"""
        f = np.zeros_like(x)
        mask = x >= self.mu
        f[mask] = (1 / self.b) * np.exp(-(x[mask] - self.mu) / self.b)
        return f

    def sample(self, n: int = 1) -> np.ndarray:
        if self.size is None:
            return np.random.exponential(scale=self.b, size=n) + self.mu
        else:
            return np.random.exponential(scale=self.b, size=(n, self.size)) + self.mu

    def __repr__(self):
        size_str = f", size={self.size}" if self.size is not None else ""
        return f"Exponential(b={self.b}, mu={self.mu}{size_str})"


def create_exponential_noise(b: float, size: int = None) -> Union[Dist, List[Dist]]:
    """指数ノイズ分布を作成"""
    exp = Exponential(b=b, size=size)
    return exp.to_dist()


class Uniform:
    """離散一様分布

    整数区間 [low, high] 上の一様分布
    各整数値が等確率 1/(high-low+1) で発生
    """

    def __init__(self, low: int, high: int, size: Union[int, None] = None):
        """
        Args:
            low: 下限（含む）
            high: 上限（含む）
            size: ベクトルサイズ（Noneの場合はスカラー）
        """
        if low > high:
            raise ValueError(f"low must be <= high, got low={low}, high={high}")

        self.low = int(low)
        self.high = int(high)
        self.size = size

    def to_dist(self) -> Union[Dist, List[Dist]]:
        """一様分布をDist形式に変換

        Returns:
            Dist: スカラーの場合
            List[Dist]: ベクトルの場合

        Note:
            依存性追跡とu範囲情報は自動的に設定されます。
            ユーザーはこれらを意識する必要がありません。
        """
        n = self.high - self.low + 1
        prob = 1.0 / n
        atoms = [(float(i), prob) for i in range(self.low, self.high + 1)]

        if self.size is None:
            def sampler(n_samples, low=self.low, high=self.high):
                return np.random.randint(low, high + 1, (n_samples, 1)).astype(float)

            dist = Dist(
                atoms=atoms,
                sampler=sampler,
                sampler_index=0,
                skip_validation=True
            )

            # sample_funcを設定（キャッシュ対応）
            def sample_func(cache, low=self.low, high=self.high):
                _ = cache  # キャッシュは依存性追跡のため渡されるが、一様分布では使用しない
                return float(np.random.randint(low, high + 1))
            dist._sample_func = sample_func

            node = Node(op='Uniform', inputs=[], dependencies=set(dist.dependencies))
            dist.node = node

            # 一様分布の場合、各値がどのu範囲で発生するかを自動設定
            # これにより、truncated_geometricなどで条件付き確率を正しく計算できる
            dist._u_ranges = {}
            for i in range(self.low, self.high + 1):
                # 各値iは、u=iの時にのみ発生（離散一様分布）
                # ただし、実際の使用では範囲として扱う
                dist._u_ranges[float(i)] = (i, i)

            # 一様分布全体の範囲も記録
            dist._u_low = self.low
            dist._u_high = self.high

            dist.normalize()
            return dist
        else:
            dists = []
            for _ in range(self.size):
                def sampler(n_samples, low=self.low, high=self.high):
                    return np.random.randint(low, high + 1, (n_samples, 1)).astype(float)

                dist = Dist(
                    atoms=atoms,
                    sampler=sampler,
                    sampler_index=0,
                    skip_validation=True
                )

                def sample_func_vec(cache, low=self.low, high=self.high):
                    _ = cache  # キャッシュは依存性追跡のため渡されるが、一様分布では使用しない
                    return float(np.random.randint(low, high + 1))
                dist._sample_func = sample_func_vec

                node = Node(op='Uniform', inputs=[], dependencies=set(dist.dependencies))
                dist.node = node

                # ベクトル版も同様にu範囲情報を設定
                dist._u_ranges = {}
                for i in range(self.low, self.high + 1):
                    dist._u_ranges[float(i)] = (i, i)
                dist._u_low = self.low
                dist._u_high = self.high

                dist.normalize()
                dists.append(dist)
            return dists

    def sample(self, n: int = 1) -> np.ndarray:
        """サンプリング（テスト用）"""
        if self.size is None:
            return np.random.randint(self.low, self.high + 1, n).astype(float)
        else:
            return np.random.randint(self.low, self.high + 1, (n, self.size)).astype(float)

    def __repr__(self):
        size_str = f", size={self.size}" if self.size is not None else ""
        return f"Uniform(low={self.low}, high={self.high}{size_str})"


def create_uniform_noise(low: int, high: int, size: int = None) -> Union[Dist, List[Dist]]:
    """便利関数：離散一様分布を作成

    Args:
        low: 下限（含む）
        high: 上限（含む）
        size: ベクトルサイズ（Noneの場合はスカラー）

    Returns:
        Dist: スカラーの場合
        List[Dist]: ベクトルの場合
    """
    uniform = Uniform(low=low, high=high, size=size)
    return uniform.to_dist()
