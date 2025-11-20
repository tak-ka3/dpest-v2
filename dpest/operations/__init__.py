"""
演算操作モジュール

確率分布に対する様々な演算を提供します。
"""

from .operations import Add, Affine, add, affine
from .max_op import Max, Min, max_op, min_op
from .argmax_op import Argmax, argmax
from .prefix_sum_op import PrefixSum, prefix_sum_distributions
from .sample_op import Sampled, sampled_distribution
from .compare_op import Compare, geq
from .branch_op import Branch, branch, Condition, NAN, PAD
from .vector_ops import vector_add, vector_argmax, vector_max, vector_min

__all__ = [
    # 基本演算
    'Add', 'Affine', 'add', 'affine',
    # 最大・最小演算
    'Max', 'Min', 'max_op', 'min_op',
    # Argmax演算
    'Argmax', 'argmax',
    # その他の演算
    'PrefixSum', 'prefix_sum_distributions',
    'Sampled', 'sampled_distribution',
    # 比較演算・条件分岐（内部使用）
    'Compare', 'Condition', 'geq',
    # センチネル値
    'NAN', 'PAD',
    # Branch演算（依存関係自動追跡、条件分岐統一インターフェース）
    'Branch', 'branch',
    # ベクトル演算
    'vector_add', 'vector_argmax', 'vector_max', 'vector_min',
]
