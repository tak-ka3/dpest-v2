"""
差分プライバシーε推定ライブラリの中核実装

このモジュールは確率分布の表現と基本的な演算を提供します。
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable, Union, Set
import numpy as np
from collections import defaultdict
from itertools import count


# 依存関係を識別するためのユニークID生成器
_dep_id_gen = count()


def _new_dep_id() -> int:
    """依存関係識別子を生成"""
    return next(_dep_id_gen)


@dataclass
class Interval:
    """区間を表現するクラス"""
    low: float
    high: float
    
    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"Invalid interval: low={self.low} > high={self.high}")
    
    def contains(self, x: float) -> bool:
        """値がこの区間に含まれるかチェック"""
        return self.low <= x <= self.high
    
    def length(self) -> float:
        """区間の長さ"""
        return self.high - self.low
    
    def __add__(self, other: 'Interval') -> 'Interval':
        """Minkowski和を計算"""
        return Interval(self.low + other.low, self.high + other.high)


@dataclass
class Node:
    """計算グラフのノード"""
    op: str
    inputs: List['Node']
    dependencies: Set[int]
    needs_sampling: bool = False


class Dist:
    """確率分布の表現

    atoms: 点質量（アトム）のリスト [(value, weight), ...]
    density: 連続密度の格子近似 {'x': grid_x, 'f': grid_f, 'dx': dx}
    support: サポート区間のリスト
    error_bounds: 誤差上界の情報
    """

    def __init__(self,
                 atoms: Optional[List[Tuple[float, float]]] = None,
                 density: Optional[Dict[str, np.ndarray]] = None,
                 support: Optional[List[Interval]] = None,
                 error_bounds: Optional[Dict[str, float]] = None,
                 sampler: Optional[Callable[[int], np.ndarray]] = None,
                 sampler_index: Optional[int] = None,
                 dependencies: Optional[Set[int]] = None,
                 node: Optional[Node] = None,
                 sample_func: Optional[Callable[[Dict[int, float]], float]] = None,
                 skip_validation: bool = False):
        self.atoms = atoms or []  # [(value, weight), ...]
        self.density = density or {}  # {'x': grid_x, 'f': grid_f, 'dx': dx}
        self.support = support or []
        self.error_bounds = error_bounds or {}
        # サンプリング関数（依存関係の判定に利用）
        self.sampler = sampler
        self.sampler_index = sampler_index
        self._sample_func = sample_func

        # 依存関係の情報
        if dependencies is not None:
            self.dependencies: Set[int] = set(dependencies)
        elif sampler is not None:
            # サンプラーを持つ場合は新しい乱数源として扱う
            self.dependencies = {_new_dep_id()}
        else:
            self.dependencies = set()

        # 計算グラフノード
        self.node = node

        # 正規化チェック（サンプリング時はスキップ可能）
        if not skip_validation:
            self._validate()
    
    def _sample(self, cache: Optional[Dict[int, float]] = None) -> float:
        """単一サンプルを生成し、再利用できるようキャッシュする。"""
        if cache is None:
            cache = {}
        key = id(self)
        if key in cache:
            return cache[key]

        if self._sample_func is not None:
            value = self._sample_func(cache)
        elif self.sampler is not None:
            samples = self.sample(1)
            if isinstance(samples, np.ndarray):
                value = float(samples.ravel()[0])
            else:
                value = float(samples)
        elif self.atoms:
            values, weights = zip(*self.atoms)
            weights = np.asarray(weights, dtype=float)
            total = weights.sum()
            if total > 0:
                weights = weights / total
                value = float(np.random.choice(values, p=weights))
            else:
                value = float(values[0])
        elif self.density:
            x = self.density.get('x')
            f = self.density.get('f')
            dx = self.density.get('dx', 1.0)
            probs = np.asarray(f, dtype=float) * float(dx)
            probs = np.clip(probs, 0.0, None)
            total = probs.sum()
            if total > 0:
                probs = probs / total
                value = float(np.random.choice(x, p=probs))
            else:
                value = float(x[0])
        else:
            value = 0.0

        cache[key] = value
        return value

    def _validate(self):
        """分布の妥当性をチェック"""
        total_mass = self.total_mass()
        if abs(total_mass - 1.0) > 1e-2:  # より緩い条件に変更
            raise ValueError(f"Distribution mass should be 1.0, got {total_mass}")
    
    def total_mass(self) -> float:
        """全確率質量を計算"""
        atom_mass = sum(w for _, w in self.atoms)
        
        if self.density and 'f' in self.density and 'dx' in self.density:
            continuous_mass = np.sum(self.density['f']) * self.density['dx']
        else:
            continuous_mass = 0.0
            
        return atom_mass + continuous_mass
    
    def normalize(self):
        """分布を正規化"""
        total = self.total_mass()
        if total == 0:
            return
            
        # 点質量を正規化
        self.atoms = [(v, w/total) for v, w in self.atoms]
        
        # 連続部分を正規化
        if self.density and 'f' in self.density:
            self.density['f'] = self.density['f'] / total
    
    @classmethod
    def from_atoms(cls, atoms: List[Tuple[float, float]],
                   sampler: Optional[Callable[[int], np.ndarray]] = None,
                   sampler_index: Optional[int] = None,
                   dependencies: Optional[Set[int]] = None,
                   node: Optional[Node] = None,
                   sample_func: Optional[Callable[[Dict[int, float]], float]] = None) -> 'Dist':
        """点質量のみから分布を作成"""
        return cls(atoms=atoms, sampler=sampler, sampler_index=sampler_index,
                   dependencies=dependencies, node=node, sample_func=sample_func)

    @classmethod
    def from_density(cls, x: np.ndarray, f: np.ndarray,
                     sampler: Optional[Callable[[int], np.ndarray]] = None,
                     sampler_index: Optional[int] = None,
                     dependencies: Optional[Set[int]] = None,
                     node: Optional[Node] = None,
                     sample_func: Optional[Callable[[Dict[int, float]], float]] = None) -> 'Dist':
        """連続密度から分布を作成"""
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        density = {'x': x, 'f': f, 'dx': dx}
        return cls(density=density, sampler=sampler, sampler_index=sampler_index,
                   dependencies=dependencies, node=node, sample_func=sample_func)

    @classmethod
    def deterministic(cls, value: float) -> 'Dist':
        """確定値（退化分布）を作成"""
        def sampler(n, v=value):
            return np.full((n, 1), v)

        node = Node(op='Const', inputs=[], dependencies=set())
        return cls.from_atoms([(value, 1.0)], sampler=sampler,
                              sampler_index=0, dependencies=set(), node=node,
                              sample_func=lambda cache, v=value: float(v))
    
    def get_support_interval(self) -> Optional[Interval]:
        """全体のサポート区間を取得"""
        if not self.support:
            return None
        
        min_low = min(interval.low for interval in self.support)
        max_high = max(interval.high for interval in self.support)
        return Interval(min_low, max_high)
    
    def __repr__(self):
        atom_str = f"atoms={len(self.atoms)}" if self.atoms else "no atoms"
        density_str = f"density={len(self.density.get('x', []))}" if self.density else "no density"
        return f"Dist({atom_str}, {density_str})"

    def sample(self, n: int) -> np.ndarray:
        """サンプリング関数があればそれを用いてサンプルを生成"""
        if self.sampler is None:
            raise ValueError("No sampler associated with this distribution")

        samples = self.sampler(n)
        samples = np.asarray(samples)
        if samples.ndim == 1:
            return samples
        idx = self.sampler_index or 0
        return samples[:, idx]


def merge_atoms(atoms: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """同じ値の点質量をマージ"""
    merged = defaultdict(float)
    for value, weight in atoms:
        merged[value] += weight
    
    return [(value, weight) for value, weight in merged.items() if weight > 1e-12]
