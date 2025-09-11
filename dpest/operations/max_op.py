"""
Max演算の実装

確率分布のリストからmax分布を計算します。
DESIGN.mdの仕様に従い、以下の公式を実装：
- Max: F_max(z) = ∏_{i=1}^k F_i(z)
- 確率密度: f_max(z) = ∑_i f_i(z) ∏_{j≠i} F_j(z)
"""

import numpy as np
from typing import List, Optional
from scipy import integrate
from ..core import Dist, Interval


class Max:
    """Max演算: Z = max(X1, X2, ..., Xk)"""
    
    @staticmethod
    def apply(distributions: List[Dist], joint_samples: Optional[np.ndarray] = None) -> Dist:
        """
        分布のリストからmax分布を計算

        Args:
            distributions: 各要素の分布のリスト
            joint_samples: 依存関係のある入力を扱うためのサンプル行列
                (n, k) 形状。提供された場合はサンプルから分布を近似する。

        Returns:
            maxの分布
        """
        if not distributions and joint_samples is None:
            raise ValueError("Empty distribution list")

        if joint_samples is not None:
            max_samples = np.max(joint_samples, axis=1)
            hist, bin_edges = np.histogram(max_samples, bins=100, density=True)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            dx = bin_edges[1] - bin_edges[0]
            result = Dist(density={'x': centers, 'f': hist, 'dx': dx})
            result.normalize()
            return result

        if len(distributions) == 1:
            return distributions[0]

        # 離散分布の場合の簡易処理
        if all(not dist.density for dist in distributions):
            return Max._compute_discrete_max(distributions)

        # 連続分布を含む場合
        return Max._compute_continuous_max(distributions)
    
    @staticmethod
    def _compute_discrete_max(distributions: List[Dist]) -> Dist:
        """離散分布のみの場合のmax計算"""
        # 全ての可能な値の組み合わせを生成
        all_combinations = []
        
        def generate_combinations(dist_idx: int, current_values: List[float], current_prob: float):
            if dist_idx == len(distributions):
                max_val = max(current_values)
                all_combinations.append((max_val, current_prob))
                return
            
            for value, prob in distributions[dist_idx].atoms:
                generate_combinations(
                    dist_idx + 1, 
                    current_values + [value], 
                    current_prob * prob
                )
        
        generate_combinations(0, [], 1.0)
        
        # 同じ最大値をまとめる
        max_probs = {}
        for max_val, prob in all_combinations:
            if max_val in max_probs:
                max_probs[max_val] += prob
            else:
                max_probs[max_val] = prob
        
        atoms = [(val, prob) for val, prob in max_probs.items()]
        return Dist.from_atoms(atoms)
    
    @staticmethod
    def _compute_continuous_max(distributions: List[Dist]) -> Dist:
        """連続分布を含む場合のmax計算"""
        # 全体のサポート範囲を決定
        all_supports = []
        for dist in distributions:
            if dist.density and 'x' in dist.density:
                x_grid = dist.density['x']
                all_supports.extend([x_grid.min(), x_grid.max()])
            if dist.atoms:
                for val, _ in dist.atoms:
                    all_supports.append(val)
        
        if not all_supports:
            raise ValueError("No support found in distributions")
        
        min_support = min(all_supports)
        max_support = max(all_supports)
        
        # 統一格子を作成
        n_grid = 1000  # 格子点数
        x_grid = np.linspace(min_support, max_support, n_grid)
        dx = (max_support - min_support) / (n_grid - 1)
        
        # f_max(z) = ∑_i f_i(z) ∏_{j≠i} F_j(z) を計算
        f_max = np.zeros(n_grid)
        
        for i in range(len(distributions)):
            # i番目の分布の密度を取得
            f_i = Max._get_density_on_grid(distributions[i], x_grid)
            
            # 他の分布のCDFの積を計算
            cdf_product = np.ones(n_grid)
            for j in range(len(distributions)):
                if i != j:
                    cdf_j = Max._get_cdf_on_grid(distributions[j], x_grid)
                    cdf_product *= cdf_j
            
            # f_i(z) * ∏_{j≠i} F_j(z) を加算
            f_max += f_i * cdf_product
        
        # normalize density before creating Dist to avoid mass errors
        f_max = f_max / (np.sum(f_max) * dx)
        result = Dist.from_density(x_grid, f_max)
        return result
    
    @staticmethod
    def _get_density_on_grid(dist: Dist, x_grid: np.ndarray) -> np.ndarray:
        """分布の密度を格子上で取得"""
        density = np.zeros_like(x_grid)
        
        # 連続部分
        if dist.density and 'x' in dist.density:
            from scipy import interpolate
            dist_x = dist.density['x']
            dist_f = dist.density['f']
            
            # 補間して格子上の値を取得
            f_interp = interpolate.interp1d(
                dist_x, dist_f, bounds_error=False, fill_value=0.0
            )
            density += f_interp(x_grid)
        
        # 点質量は無視（密度関数としては0）
        return density
    
    @staticmethod
    def _get_cdf_on_grid(dist: Dist, x_grid: np.ndarray) -> np.ndarray:
        """分布のCDFを格子上で取得"""
        cdf = np.zeros_like(x_grid)
        
        # 連続部分のCDF
        if dist.density and 'x' in dist.density:
            dist_x = dist.density['x']
            dist_f = dist.density['f']
            
            for i, x in enumerate(x_grid):
                mask = dist_x <= x
                if np.any(mask):
                    cdf[i] = np.trapz(dist_f[mask], dist_x[mask])
        
        # 点質量の寄与
        if dist.atoms:
            for atom_value, atom_weight in dist.atoms:
                mask = x_grid >= atom_value
                cdf[mask] += atom_weight
        
        return np.clip(cdf, 0, 1)


class Min:
    """Min演算: Z = min(X1, X2, ..., Xk)"""

    @staticmethod
    def apply(distributions: List[Dist], joint_samples: Optional[np.ndarray] = None) -> Dist:
        """
        分布のリストからmin分布を計算
        公式: 1 - F_min(z) = ∏_i (1 - F_i(z))

        Args:
            distributions: 各要素の分布のリスト
            joint_samples: 依存する入力のサンプル行列 (n, k)

        """
        if not distributions and joint_samples is None:
            raise ValueError("Empty distribution list")

        if joint_samples is not None:
            min_samples = np.min(joint_samples, axis=1)
            hist, bin_edges = np.histogram(min_samples, bins=100, density=True)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            dx = bin_edges[1] - bin_edges[0]
            result = Dist(density={'x': centers, 'f': hist, 'dx': dx})
            result.normalize()
            return result

        if len(distributions) == 1:
            return distributions[0]

        # 離散分布の場合の簡易処理
        if all(not dist.density for dist in distributions):
            return Min._compute_discrete_min(distributions)

        # 連続分布を含む場合
        return Min._compute_continuous_min(distributions)
    
    @staticmethod
    def _compute_discrete_min(distributions: List[Dist]) -> Dist:
        """離散分布のみの場合のmin計算"""
        # 全ての可能な値の組み合わせを生成
        all_combinations = []
        
        def generate_combinations(dist_idx: int, current_values: List[float], current_prob: float):
            if dist_idx == len(distributions):
                min_val = min(current_values)
                all_combinations.append((min_val, current_prob))
                return
            
            for value, prob in distributions[dist_idx].atoms:
                generate_combinations(
                    dist_idx + 1, 
                    current_values + [value], 
                    current_prob * prob
                )
        
        generate_combinations(0, [], 1.0)
        
        # 同じ最小値をまとめる
        min_probs = {}
        for min_val, prob in all_combinations:
            if min_val in min_probs:
                min_probs[min_val] += prob
            else:
                min_probs[min_val] = prob
        
        atoms = [(val, prob) for val, prob in min_probs.items()]
        return Dist.from_atoms(atoms)
    
    @staticmethod
    def _compute_continuous_min(distributions: List[Dist]) -> Dist:
        """連続分布を含む場合のmin計算"""
        # 全体のサポート範囲を決定
        all_supports = []
        for dist in distributions:
            if dist.density and 'x' in dist.density:
                x_grid = dist.density['x']
                all_supports.extend([x_grid.min(), x_grid.max()])
            if dist.atoms:
                for val, _ in dist.atoms:
                    all_supports.append(val)
        
        if not all_supports:
            raise ValueError("No support found in distributions")
        
        min_support = min(all_supports)
        max_support = max(all_supports)
        
        # 統一格子を作成
        n_grid = 1000
        x_grid = np.linspace(min_support, max_support, n_grid)
        
        # F_min(z) = 1 - ∏_i (1 - F_i(z)) を計算
        f_min_cdf = np.ones(n_grid)
        
        for dist in distributions:
            cdf_i = Max._get_cdf_on_grid(dist, x_grid)
            f_min_cdf *= (1 - cdf_i)
        
        f_min_cdf = 1 - f_min_cdf
        
        # 密度を数値微分で計算
        f_min_density = np.gradient(f_min_cdf, x_grid)
        f_min_density = np.maximum(f_min_density, 0)  # 負の値を0にクリップ
        
        result = Dist.from_density(x_grid, f_min_density)
        result.normalize()
        return result


def max_distribution(distributions: List[Dist], joint_samples: Optional[np.ndarray] = None) -> Dist:
    """便利関数：max分布を計算"""
    return Max.apply(distributions, joint_samples=joint_samples)


def min_distribution(distributions: List[Dist], joint_samples: Optional[np.ndarray] = None) -> Dist:
    """便利関数：min分布を計算"""
    return Min.apply(distributions, joint_samples=joint_samples)