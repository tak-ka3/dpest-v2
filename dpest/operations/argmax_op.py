"""
Argmax演算の実装

確率分布のリストからargmax分布を計算します。
"""

import numpy as np
from typing import List, Optional
from scipy import integrate
from ..core import Dist, Node


class Argmax:
    """Argmax演算: P(argmax=i) = ∫ f_i(x) ∏_{j≠i} F_j(x) dx"""
    
    @staticmethod
    def apply(distributions: List[Dist], joint_samples: Optional[np.ndarray] = None) -> Dist:
        """
        分布のリストからargmax分布を計算

        Args:
            distributions: 各要素の分布のリスト
            joint_samples: 依存する入力サンプル (n, k)。指定された場合は
                サンプルからargmax分布を近似する。

        Returns:
            argmaxの離散分布（各インデックスの確率）
        """
        union_deps = set().union(*[getattr(d, 'dependencies', set()) for d in distributions])

        if joint_samples is None:
            # 依存関係チェック
            dependent = False
            for i in range(len(distributions)):
                for j in range(i + 1, len(distributions)):
                    if (getattr(distributions[i], 'dependencies', set()) &
                            getattr(distributions[j], 'dependencies', set())) or \
                       (getattr(distributions[i], 'sampler', None) is not None and
                        distributions[i].sampler is distributions[j].sampler):
                        dependent = True
                        break
                if dependent:
                    break

            if dependent:
                samplers = [getattr(d, 'sampler', None) for d in distributions]
                if any(s is None for s in samplers) or len({id(s) for s in samplers}) != 1:
                    raise ValueError("Dependent inputs require joint samples or shared sampler")
                base_sampler = samplers[0]
                indices = [d.sampler_index or 0 for d in distributions]
                samples = base_sampler(1000)
                samples = np.asarray(samples)
                if samples.ndim == 1:
                    samples = samples.reshape(-1, 1)
                joint_samples = np.column_stack([samples[:, idx] for idx in indices])

        if joint_samples is not None:
            indices = np.argmax(joint_samples, axis=1)
            unique, counts = np.unique(indices, return_counts=True)
            total = len(indices)
            atoms = [(int(idx), cnt/total) for idx, cnt in zip(unique, counts)]
            node = Node(op='Argmax', inputs=[getattr(d, 'node', None) for d in distributions],
                        dependencies=union_deps)
            return Dist.from_atoms(atoms, dependencies=union_deps, node=node)

        n = len(distributions)
        if n == 0:
            raise ValueError("Empty distribution list")

        argmax_probs = []

        for i in range(n):
            # P(argmax=i) を計算
            prob_i = Argmax._compute_argmax_prob(distributions, i)
            argmax_probs.append((i, prob_i))

        # 正規化
        total_prob = sum(prob for _, prob in argmax_probs)
        if total_prob > 0:
            argmax_probs = [(idx, prob/total_prob) for idx, prob in argmax_probs]

        node = Node(op='Argmax', inputs=[getattr(d, 'node', None) for d in distributions],
                    dependencies=union_deps)
        return Dist.from_atoms(argmax_probs, dependencies=union_deps, node=node)
    
    @staticmethod
    def _compute_argmax_prob(distributions: List[Dist], target_idx: int) -> float:
        """
        P(argmax=target_idx) = ∫ f_target(x) ∏_{j≠target} F_j(x) dx を計算
        つまり、target_idxにおける値が最大になり、他の確率変数の値はそれ以下になる確率を考える
        """
        target_dist = distributions[target_idx]
        other_dists = [distributions[j] for j in range(len(distributions)) if j != target_idx]
        
        # 簡略化: 連続分布のみを仮定し、格子上で数値積分
        if not target_dist.density or 'x' not in target_dist.density:
            # 点質量のみの場合の簡易処理
            return Argmax._compute_discrete_argmax_prob(distributions, target_idx)
        
        x_grid = target_dist.density['x']
        f_target = target_dist.density['f']
        dx = target_dist.density['dx']
        
        # 各点での累積分布関数の積を計算
        integrand = f_target.copy()
        
        # ∏_{j≠target} F_j(x) を計算
        for other_dist in other_dists:
            if other_dist.density and 'x' in other_dist.density:
                # CDFを数値的に計算（つまりこの時、比較する二つの確率変数がどちらも確率密度を持つことになる）
                cdf_values = Argmax._compute_cdf_on_grid(other_dist, x_grid)
                integrand *= cdf_values
            else:
                # 点質量の場合のCDF
                cdf_values = Argmax._compute_discrete_cdf_on_grid(other_dist, x_grid)
                integrand *= cdf_values
        
        # integrand (=f_target(x) ∏_{j≠target} F_j(x))をx_gridで数値積分
        return np.trapz(integrand, x_grid)
    
    @staticmethod
    def _compute_discrete_argmax_prob(distributions: List[Dist], target_idx: int) -> float:
        """点質量のみの場合のargmax確率計算"""
        if not distributions[target_idx].atoms:
            return 0.0
        
        total_prob = 0.0
        
        for target_value, target_weight in distributions[target_idx].atoms:
            # この値が最大になる確率
            prob_max = target_weight
            
            # 他の分布でこの値以下になる確率
            for j, other_dist in enumerate(distributions):
                if j == target_idx:
                    continue
                
                prob_le = 0.0
                if other_dist.atoms:
                    for other_value, other_weight in other_dist.atoms:
                        if other_value <= target_value:
                            prob_le += other_weight
                
                prob_max *= prob_le
            
            total_prob += prob_max
        
        return total_prob
    
    @staticmethod
    def _compute_cdf_on_grid(dist: Dist, x_grid: np.ndarray) -> np.ndarray:
        """格子点上でCDFを計算"""
        if not dist.density or 'x' not in dist.density:
            return np.ones_like(x_grid)
        
        dist_x = dist.density['x']
        dist_f = dist.density['f']
        dx = dist.density['dx']
        
        cdf_values = np.zeros_like(x_grid)
        
        for i, x in enumerate(x_grid):
            # 台形近似（trapz）により、x以下の確率質量を計算
            mask = dist_x <= x
            if np.any(mask):
                cdf_values[i] = np.trapz(dist_f[mask], dist_x[mask])
        
        # 点質量がある場合は追加
        if dist.atoms:
            for atom_value, atom_weight in dist.atoms:
                mask = x_grid >= atom_value
                cdf_values[mask] += atom_weight
        
        # cdf_valuesはx_gridに対応する確率質量を表す
        return np.clip(cdf_values, 0, 1)
    
    @staticmethod
    def _compute_discrete_cdf_on_grid(dist: Dist, x_grid: np.ndarray) -> np.ndarray:
        """点質量分布のCDFを格子上で計算"""
        cdf_values = np.zeros_like(x_grid)
        
        if dist.atoms:
            for atom_value, atom_weight in dist.atoms:
                mask = x_grid >= atom_value
                cdf_values[mask] += atom_weight
        
        return np.clip(cdf_values, 0, 1)


def argmax_distribution(distributions: List[Dist], joint_samples: Optional[np.ndarray] = None) -> Dist:
    """便利関数：argmax分布を計算"""
    return Argmax.apply(distributions, joint_samples=joint_samples)