"""
差分プライバシーε推定ライブラリ

確率分布の計算と変換を通じて、アルゴリズムのプライバシーパラメータを推定します。
"""

from core import Dist, Interval
from noise import Laplace, create_laplace_noise
from operations import Add, Affine, add_distributions, affine_transform
from argmax import Argmax, argmax_distribution
from engine import Engine, compile, AlgorithmBuilder, Laplace_dist, vector_argmax

__version__ = "0.1.0"

__all__ = [
    # 中核クラス
    'Dist', 'Interval',
    # ノイズ機構
    'Laplace', 'create_laplace_noise', 'Laplace_dist',
    # 演算
    'Add', 'Affine', 'add_distributions', 'affine_transform',
    'Argmax', 'argmax_distribution', 'vector_argmax',
    # エンジン
    'Engine', 'compile', 'AlgorithmBuilder',
]