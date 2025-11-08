"""
差分プライバシーε推定ライブラリ

確率分布の計算と変換を通じて、アルゴリズムのプライバシーパラメータを推定します。
"""

from .core import Dist, Interval
from .noise import Laplace, Exponential, create_laplace_noise, create_exponential_noise
from .operations import (
    Add, Affine, add, affine,
    Max, Min, max_op, min_op,
    Argmax, argmax,
    Sampled, sampled_distribution,
    geq, condition, MUX, mux, NAN, PAD,
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
    'Laplace', 'Exponential', 'create_laplace_noise', 'create_exponential_noise',
    'Laplace_dist', 'Exponential_dist',
    # 演算
    'Add', 'Affine', 'add', 'affine',
    'Argmax', 'argmax',
    'Max', 'Min', 'max_op', 'min_op',
    'Sampled', 'sampled_distribution',
    'GE', 'geq', 'condition', 'MUX', 'mux', 'NAN', 'PAD',
    # ベクトル演算
    'vector_add', 'vector_argmax', 'vector_max', 'vector_min',
    # エンジン
    'Engine', 'compile', 'AlgorithmBuilder',
]
