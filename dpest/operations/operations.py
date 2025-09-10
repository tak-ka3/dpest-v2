"""
確率分布に対する演算操作の実装

AddやAffineなどの基本的な確率分布演算を実装します。
"""

import numpy as np
from typing import List, Union, Optional
from scipy import interpolate
from ..core import Dist, Interval, merge_atoms


class Add:
    """加法演算: Z = X + Y"""

    @staticmethod
    def apply(x_dist: Dist, y_dist: Dist, joint_samples: Optional[np.ndarray] = None,
              n_samples: int = 1000) -> Dist:
        """
        二つの分布の和を計算する。

        Args:
            x_dist, y_dist: 足し合わせる分布。通常は独立を仮定する。
            joint_samples: (n,2) 形状のサンプル行列。入力が独立でない場合に
                共通サンプルから分布を近似するために使用する。

        Returns:
            Z = X + Y の分布
        """

        base_sampler = None
        x_idx = y_idx = None

        # 入力分布が同一のサンプラーを共有している場合、依存とみなす
        if joint_samples is None and getattr(x_dist, 'sampler', None) is not None \
           and getattr(y_dist, 'sampler', None) is not None \
           and x_dist.sampler is y_dist.sampler:
            base_sampler = x_dist.sampler
            x_idx = x_dist.sampler_index or 0
            y_idx = y_dist.sampler_index or 0
            samples = base_sampler(n_samples)
            samples = np.asarray(samples)
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)
            joint_samples = np.column_stack((samples[:, x_idx], samples[:, y_idx]))

        # 依存する入力への対処（サンプルベースの近似）
        if joint_samples is not None:
            sums = joint_samples[:, 0] + joint_samples[:, 1]
            hist, bin_edges = np.histogram(sums, bins=100, density=True)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            dx = bin_edges[1] - bin_edges[0]
            result = Dist(density={'x': centers, 'f': hist, 'dx': dx})
            result.normalize()
            # サンプラーが存在する場合は結果にも伝播
            if base_sampler is not None:
                def sampler(n, bs=base_sampler, xi=x_idx, yi=y_idx):
                    data = bs(n)
                    data = np.asarray(data)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    return data[:, xi] + data[:, yi]

                result.sampler = sampler
            return result

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
                    shifted_x = y_x + x_val # [y_x_1 + x_val, y_x_2 + x_val, ...]
                    shifted_f = y_f * x_weight # [y_f_1 * x_weight, y_f_2 * x_weight, ...] 確率密度に確率という重みをかける
                    all_shifted_x.extend(shifted_x) # appendではなくextendなので、リストを1次元で拡張
                    all_shifted_f.extend(shifted_f)
                
                # 確率変数の値の最小値と最大値を等間隔でn_grid個だけグリッド化
                if all_shifted_x:
                    # 統一グリッドで補間
                    min_x = min(all_shifted_x)
                    max_x = max(all_shifted_x)
                    n_grid = len(y_x)
                    unified_x = np.linspace(min_x, max_x, n_grid)
                    unified_f = np.zeros(n_grid)
                    
                    # 各シフト分布を統一グリッドに補間して加算
                    for x_val, x_weight in x_dist.atoms:
                        shifted_x = y_x + x_val # [y_x_1 + x_val, y_x_2 + x_val, ...]
                        shifted_f = y_f * x_weight # [y_f_1 * x_weight, y_f_2 * x_weight, ...]
                        f_interp = interpolate.interp1d(shifted_x, shifted_f, 
                                                       bounds_error=False, fill_value=0.0)
                        unified_f += f_interp(unified_x) # 各点での確率密度を加算
                    
                    result_density = {'x': unified_x, 'f': unified_f, 'dx': dx}
        
        # 連続+離散の処理（対称性）
        if y_dist.atoms and x_dist.density:
            return Add.apply(y_dist, x_dist)  # 順序を入れ替えて再帰
        
        # 連続+連続の処理（FFTによる畳み込み）
        if x_dist.density and y_dist.density and 'x' in x_dist.density and 'x' in y_dist.density:
            result_density = Add._convolve_continuous(x_dist.density, y_dist.density)
        
        # サポートの計算（Minkowski和）
        # 結果分布 Z = X + Y がゼロでない値を取りうる範囲を事前に計算している
        x_support = x_dist.get_support_interval() # Interval(low_1, high_1) or None
        y_support = y_dist.get_support_interval() # Interval(low_2, high_2) or None
        if x_support and y_support:
            result_support = [x_support + y_support] # Interval(low_1 + low_2, high_1 + high_2)
        
        # 点質量をマージ
        result_atoms = merge_atoms(result_atoms)
        
        result = Dist(atoms=result_atoms, density=result_density, support=result_support)
        result.normalize()

        # サンプリング関数の伝播
        if getattr(x_dist, 'sampler', None) is not None and getattr(y_dist, 'sampler', None) is not None:
            if x_dist.sampler is y_dist.sampler:
                base_sampler = x_dist.sampler
                x_idx = x_dist.sampler_index or 0
                y_idx = y_dist.sampler_index or 0

                def sampler(n, bs=base_sampler, xi=x_idx, yi=y_idx):
                    data = bs(n)
                    data = np.asarray(data)
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    return data[:, xi] + data[:, yi]
            else:
                def sampler(n, xs=x_dist.sample, ys=y_dist.sample):
                    return xs(n) + ys(n)
            result.sampler = sampler

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


def add_distributions(x_dist: Dist, y_dist: Union[Dist, List[Dist]],
                     joint_samples: Optional[np.ndarray] = None,
                     n_samples: int = 1000) -> Union[Dist, List[Dist]]:
    """便利関数：分布の加法"""
    if isinstance(y_dist, list):
        if joint_samples is not None:
            raise ValueError("joint_samples はリスト入力ではサポートされていません")
        return [Add.apply(x_dist, y, n_samples=n_samples) for y in y_dist]
    else:
        return Add.apply(x_dist, y_dist, joint_samples=joint_samples, n_samples=n_samples)


def affine_transform(x_dist: Dist, a: float, b: float = 0.0) -> Dist:
    """便利関数：アフィン変換"""
    return Affine.apply(x_dist, a, b)