"""
差分プライバシーアルゴリズムの宣言的実装

このモジュールでは、compile()を使用する分布ベースのアルゴリズム実装を提供します。
"""

from .svt1 import svt1
from .svt2 import svt2
from .svt3 import svt3
from .svt4 import svt4
from .svt5 import svt5
from .svt6 import svt6
from .numerical_svt import numerical_svt
from .noisy_hist1 import noisy_hist1
from .noisy_hist2 import noisy_hist2
from .report_noisy_max1 import report_noisy_max1
from .report_noisy_max2 import report_noisy_max2
from .report_noisy_max3 import report_noisy_max3
from .report_noisy_max4 import report_noisy_max4
from .laplace_vec import laplace_vec
from .laplace_parallel import laplace_parallel
from .one_time_rappor import one_time_rappor
from .rappor import rappor
from .noisy_max_sum import noisy_max_sum
from .truncated_geometric import truncated_geometric
from .registry import get_registered_dist_functions

__all__ = [
    'svt1', 'svt2', 'svt3', 'svt4', 'svt5', 'svt6', 'numerical_svt',
    'noisy_hist1', 'noisy_hist2',
    'report_noisy_max1', 'report_noisy_max2',
    'report_noisy_max3', 'report_noisy_max4',
    'laplace_vec', 'laplace_parallel',
    'one_time_rappor', 'rappor',
    'noisy_max_sum',
    'truncated_geometric',
]

# Automatically expose all auto-generated dist functions (e.g., svt1_dist)
_auto_dist_functions = get_registered_dist_functions()
globals().update(_auto_dist_functions)
__all__.extend(sorted(_auto_dist_functions.keys()))
