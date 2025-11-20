"""
差分プライバシーε推定ライブラリ

確率分布の計算と変換を通じて、アルゴリズムのプライバシーパラメータを推定します。
"""

from .core import Dist, Interval
from .noise import Laplace, Exponential, Uniform, create_laplace_noise, create_exponential_noise, create_uniform_noise
from .operations import (
    Add, Affine, add, affine,
    Max, Min, max_op, min_op,
    Argmax, argmax,
    Sampled, sampled_distribution,
    geq, NAN, PAD,
    Branch, branch,
)
from .engine import (Engine, compile, AlgorithmBuilder, Laplace_dist, Exponential_dist,
                     vector_add, vector_argmax, vector_max, vector_min)

__version__ = "0.1.0"

# GEエイリアス (geqの別名)
GE = geq

__all__ = [
    # 中核クラス
    'Dist', 'Interval',
    # ノイズ機構
    'Laplace', 'Exponential', 'Uniform',
    'create_laplace_noise', 'create_exponential_noise', 'create_uniform_noise',
    'Laplace_dist', 'Exponential_dist',
    # 演算
    'Add', 'Affine', 'add', 'affine',
    'Argmax', 'argmax',
    'Max', 'Min', 'max_op', 'min_op',
    'Sampled', 'sampled_distribution',
    'GE', 'geq', 'NAN', 'PAD',
    'Branch', 'branch',
    # ベクトル演算
    'vector_add', 'vector_argmax', 'vector_max', 'vector_min',
    # エンジン
    'Engine', 'compile', 'AlgorithmBuilder',
]
