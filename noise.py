"""
ノイズ機構の実装

差分プライバシーで使用されるノイズ分布を実装します。
"""

import numpy as np
from typing import Union, List
from core import Dist, Interval


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
        
        dist = Dist.from_density(x, f)
        dist.support = [Interval(self.mu - support_range, self.mu + support_range)]
        
        if self.size is None:
            return dist
        else:
            # 独立なラプラス分布のリストを返す
            return [dist for _ in range(self.size)]
    
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


def create_laplace_noise(b: float, size: int = None) -> Union[Dist, List[Dist]]:
    """便利関数：ラプラスノイズ分布を作成"""
    laplace = Laplace(b=b, size=size)
    return laplace.to_dist()