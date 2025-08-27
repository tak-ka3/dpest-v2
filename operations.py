"""
確率分布に対する演算操作の実装

AddやAffineなどの基本的な確率分布演算を実装します。
"""

import numpy as np
from typing import List, Union
from scipy import interpolate
from core import Dist, Interval, merge_atoms


class Add:
    """加法演算: Z = X + Y"""
    
    @staticmethod
    def apply(x_dist: Dist, y_dist: Dist) -> Dist:
        """独立な二つの分布の和を計算"""
        result_atoms = []
        result_density = {}
        result_support = []
        
        # 離散+離散の処理
        if x_dist.atoms and y_dist.atoms:
            for x_val, x_weight in x_dist.atoms:
                for y_val, y_weight in y_dist.atoms:
                    result_atoms.append((x_val + y_val, x_weight * y_weight))
        
        # 離散+連続の処理
        if x_dist.atoms and y_dist.density:
            if 'x' in y_dist.density and 'f' in y_dist.density:
                y_x = y_dist.density['x']
                y_f = y_dist.density['f']
                dx = y_dist.density['dx']
                
                # 各点質量に対してシフトした連続分布を追加
                all_shifted_x = []
                all_shifted_f = []
                
                for x_val, x_weight in x_dist.atoms:
                    shifted_x = y_x + x_val
                    shifted_f = y_f * x_weight
                    all_shifted_x.extend(shifted_x)
                    all_shifted_f.extend(shifted_f)
                
                if all_shifted_x:
                    # 統一グリッドで補間
                    min_x = min(all_shifted_x)
                    max_x = max(all_shifted_x)
                    n_grid = len(y_x)
                    unified_x = np.linspace(min_x, max_x, n_grid)
                    unified_f = np.zeros(n_grid)
                    
                    # 各シフト分布を統一グリッドに補間して加算
                    for x_val, x_weight in x_dist.atoms:
                        shifted_x = y_x + x_val
                        shifted_f = y_f * x_weight
                        f_interp = interpolate.interp1d(shifted_x, shifted_f, 
                                                       bounds_error=False, fill_value=0.0)
                        unified_f += f_interp(unified_x)
                    
                    result_density = {'x': unified_x, 'f': unified_f, 'dx': dx}
        
        # 連続+離散の処理（対称性）
        if y_dist.atoms and x_dist.density:
            return Add.apply(y_dist, x_dist)  # 順序を入れ替えて再帰
        
        # 連続+連続の処理（FFTによる畳み込み）
        if x_dist.density and y_dist.density and 'x' in x_dist.density and 'x' in y_dist.density:
            result_density = Add._convolve_continuous(x_dist.density, y_dist.density)
        
        # サポートの計算（Minkowski和）
        x_support = x_dist.get_support_interval()
        y_support = y_dist.get_support_interval()
        if x_support and y_support:
            result_support = [x_support + y_support]
        
        # 点質量をマージ
        result_atoms = merge_atoms(result_atoms)
        
        result = Dist(atoms=result_atoms, density=result_density, support=result_support)
        result.normalize()
        return result
    
    @staticmethod
    def _convolve_continuous(x_density: dict, y_density: dict) -> dict:
        """FFTを使った連続分布の畳み込み"""
        x_grid = x_density['x']
        x_f = x_density['f']
        y_grid = y_density['x']
        y_f = y_density['f']
        
        # 格子を統一
        dx = min(x_density['dx'], y_density['dx'])
        min_x = min(x_grid[0], y_grid[0])
        max_x = max(x_grid[-1], y_grid[-1])
        
        # 新しい統一グリッドを作成
        n_points = int((max_x - min_x) / dx) + 1
        unified_x = np.linspace(min_x, max_x, n_points)
        
        # 各分布を統一グリッドに補間
        f_x_interp = interpolate.interp1d(x_grid, x_f, bounds_error=False, fill_value=0.0)
        f_y_interp = interpolate.interp1d(y_grid, y_f, bounds_error=False, fill_value=0.0)
        
        x_unified = f_x_interp(unified_x)
        y_unified = f_y_interp(unified_x)
        
        # FFTで畳み込み
        conv_result = np.convolve(x_unified, y_unified, mode='full') * dx
        
        # 結果のグリッド
        result_x = np.linspace(2*min_x, 2*max_x, len(conv_result))
        
        return {'x': result_x, 'f': conv_result, 'dx': dx}


class Affine:
    """アフィン変換: Z = aX + b"""
    
    @staticmethod
    def apply(x_dist: Dist, a: float, b: float) -> Dist:
        """アフィン変換を適用"""
        if a == 0:
            # 退化分布
            return Dist.deterministic(b)
        
        result_atoms = []
        result_density = {}
        result_support = []
        
        # 点質量の変換
        if x_dist.atoms:
            result_atoms = [(a * x_val + b, x_weight) for x_val, x_weight in x_dist.atoms]
        
        # 連続部分の変換
        if x_dist.density and 'x' in x_dist.density:
            x_grid = x_dist.density['x']
            x_f = x_dist.density['f']
            dx = x_dist.density['dx']
            
            # 変換後のグリッド
            z_grid = a * x_grid + b
            z_f = x_f / abs(a)  # ヤコビアン
            
            # 昇順にソート（aが負の場合）
            if a < 0:
                sort_idx = np.argsort(z_grid)
                z_grid = z_grid[sort_idx]
                z_f = z_f[sort_idx]
            
            result_density = {'x': z_grid, 'f': z_f, 'dx': abs(a) * dx}
        
        # サポートの変換
        if x_dist.support:
            for interval in x_dist.support:
                if a > 0:
                    new_interval = Interval(a * interval.low + b, a * interval.high + b)
                else:
                    new_interval = Interval(a * interval.high + b, a * interval.low + b)
                result_support.append(new_interval)
        
        return Dist(atoms=result_atoms, density=result_density, support=result_support)


def add_distributions(x_dist: Dist, y_dist: Union[Dist, List[Dist]]) -> Union[Dist, List[Dist]]:
    """便利関数：分布の加法"""
    if isinstance(y_dist, list):
        return [Add.apply(x_dist, y) for y in y_dist]
    else:
        return Add.apply(x_dist, y_dist)


def affine_transform(x_dist: Dist, a: float, b: float = 0.0) -> Dist:
    """便利関数：アフィン変換"""
    return Affine.apply(x_dist, a, b)