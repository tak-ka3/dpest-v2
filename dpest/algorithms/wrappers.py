"""
privacy_loss_report.pyのインターフェースに適合させるためのラッパー関数

dist_funcやjoint_dist_funcとして使用できるように、
compile()ベースのアルゴリズムをラップします。
"""

import numpy as np
from typing import List
from ..core import Dist
from ..engine import compile
from .svt1 import svt1 as svt1_impl
from .svt2 import svt2 as svt2_impl
from .svt3 import svt3 as svt3_impl
from .svt4 import svt4 as svt4_impl
from .svt5 import svt5 as svt5_impl
from .svt6 import svt6 as svt6_impl
from .numerical_svt import numerical_svt as numerical_svt_impl


def svt1_dist(data: np.ndarray, eps: float) -> List[Dist]:
    """SVT1をdist_funcインターフェースに適合させるラッパー"""
    queries = [Dist.deterministic(float(q)) for q in data]
    svt1_compiled = compile(lambda q: svt1_impl(q, eps))
    return svt1_compiled(queries)


def svt2_dist(data: np.ndarray, eps: float) -> List[Dist]:
    """SVT2をdist_funcインターフェースに適合させるラッパー"""
    queries = [Dist.deterministic(float(q)) for q in data]
    svt2_compiled = compile(lambda q: svt2_impl(q, eps))
    return svt2_compiled(queries)


def svt3_dist(data: np.ndarray, eps: float) -> List[Dist]:
    """SVT3をdist_funcインターフェースに適合させるラッパー"""
    queries = [Dist.deterministic(float(q)) for q in data]
    svt3_compiled = compile(lambda q: svt3_impl(q, eps))
    return svt3_compiled(queries)


def svt4_dist(data: np.ndarray, eps: float) -> List[Dist]:
    """SVT4をdist_funcインターフェースに適合させるラッパー"""
    queries = [Dist.deterministic(float(q)) for q in data]
    svt4_compiled = compile(lambda q: svt4_impl(q, eps))
    return svt4_compiled(queries)


def svt5_dist(data: np.ndarray, eps: float) -> List[Dist]:
    """SVT5をdist_funcインターフェースに適合させるラッパー"""
    queries = [Dist.deterministic(float(q)) for q in data]
    svt5_compiled = compile(lambda q: svt5_impl(q, eps))
    return svt5_compiled(queries)


def svt6_dist(data: np.ndarray, eps: float) -> List[Dist]:
    """SVT6をdist_funcインターフェースに適合させるラッパー"""
    queries = [Dist.deterministic(float(q)) for q in data]
    svt6_compiled = compile(lambda q: svt6_impl(q, eps))
    return svt6_compiled(queries)


def numerical_svt_dist(data: np.ndarray, eps: float) -> List[Dist]:
    """NumericalSVTをdist_funcインターフェースに適合させるラッパー"""
    queries = [Dist.deterministic(float(q)) for q in data]
    numerical_svt_compiled = compile(lambda q: numerical_svt_impl(q, eps))
    return numerical_svt_compiled(queries)
