"""
差分プライバシーε推定エンジンの実装

アルゴリズムをコンパイルして分布計算関数を生成します。
"""

from dataclasses import dataclass
from typing import Callable, Any, List, Union, Optional
from .core import Dist
from .noise import Laplace, Exponential, create_laplace_noise, create_exponential_noise
from .operations import (
    Add,
    Affine,
    add_distributions,
    affine_transform,
    Max,
    Min,
    max_distribution,
    min_distribution,
    Argmax,
    argmax_distribution,
    PrefixSum,
    prefix_sum_distributions,
    Sampled,
    sampled_distribution,
    Compare,
    Condition,
    compare_geq,
    condition_mixture,
    TruncatedGeometric,
    truncated_geometric_distribution,
)


class Engine:
    """差分プライバシーε推定エンジンのメインクラス"""

    def __init__(self):
        # 利用可能な操作のマッピング
        self.operations = {
            'Laplace': Laplace,
            'Exponential': Exponential,
            'Add': Add,
            'Affine': Affine,
            'Argmax': Argmax,
            'Max': Max,
            'Min': Min,
            'Compare': Compare,
            'Condition': Condition,
            'PrefixSum': PrefixSum,
            'Sampled': Sampled,
            'TruncatedGeometric': TruncatedGeometric,
        }

    # アルゴリズムの関数から計算グラフを作る
    def _build_computation_graph(self, algo_func: Callable):
        """与えられたアルゴリズム関数から計算グラフを生成する"""

        if not callable(algo_func):
            raise TypeError("Algorithm must be callable")

        # 現状ではアルゴリズム関数そのものを計算グラフとして扱う
        return algo_func

    # 計算グラフの最適化（シンプルな恒等最適化）
    def _optimize_graph(self, computation_graph: Callable):
        """計算グラフの最適化。現状では恒等変換を行う。"""

        if not callable(computation_graph):
            raise TypeError("Computation graph must be callable")
        return computation_graph

    @dataclass
    class ExecutionPlan:
        """出力分布計算のための実行計画"""

        mode: str
        graph: Callable
        options: Optional[dict] = None

    def _analyze_node(self, node) -> bool:
        """ノードがサンプリングを要するか解析"""
        if node is None:
            return False
        for inp in getattr(node, 'inputs', []):
            self._analyze_node(inp)
        deps = [getattr(inp, 'dependencies', set()) for inp in getattr(node, 'inputs', [])]
        # 子がサンプリング必要なら伝播
        if any(getattr(inp, 'needs_sampling', False) for inp in getattr(node, 'inputs', [])):
            node.needs_sampling = True
            return True
        # 依存関係の重なりをチェック
        for i in range(len(deps)):
            for j in range(i + 1, len(deps)):
                if deps[i] & deps[j]:
                    node.needs_sampling = True
                    return True
        node.needs_sampling = False
        return False

    def _plan_execution(self, optimized_graph: Callable, input_dist) -> "Engine.ExecutionPlan":
        """計算グラフから出力分布の計算方法を決定する"""

        result = optimized_graph(input_dist)
        self._analyze_node(getattr(result, 'node', None))
        mode = 'sampling' if getattr(result.node, 'needs_sampling', False) else 'analytic'

        return self.ExecutionPlan(mode=mode, graph=lambda _: result)

    # 最適化済み計算グラフと実行計画に基づいて出力分布を得る
    def _execute_algorithm(self, plan: "Engine.ExecutionPlan", input_dist):
        """実行計画に従ってアルゴリズムを実行し分布を得る"""

        graph = plan.graph
        if not callable(graph):
            raise ValueError("Execution plan graph is not callable")

        return graph(input_dist)

    def compile(self, algo_func: Callable) -> Callable:
        """アルゴリズム関数から最適な計算グラフを生成して分布計算関数を返す（コンパイル）"""

        computation_graph = self._build_computation_graph(algo_func)
        optimized_graph = self._optimize_graph(computation_graph)

        def distribution_func(input_data):
            """入力データから出力分布を計算"""

            # 入力データから初期分布を作成
            input_dist = self._create_input_distribution(input_data)

            # 実行計画の作成とアルゴリズム実行
            plan = self._plan_execution(optimized_graph, input_dist)
            output_dist = self._execute_algorithm(plan, input_dist)

            return output_dist

        return distribution_func

    def _create_input_distribution(self, input_data) -> Union[Dist, List[Dist]]:
        """入力データから初期分布を作成"""
        if isinstance(input_data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in input_data):
                # 数値リストの場合、各要素を確定値分布として作成
                return [Dist.deterministic(float(x)) for x in input_data]
            else:
                # 既にDistオブジェクトのリストの場合
                return input_data
        elif isinstance(input_data, (int, float)):
            # スカラー値の場合
            return Dist.deterministic(float(input_data))
        elif isinstance(input_data, Dist):
            # 既にDistオブジェクトの場合
            return input_data
        else:
            raise ValueError(f"Unsupported input data type: {type(input_data)}")


# グローバル関数として提供
_default_engine = Engine()

def compile(algo_func: Callable) -> Callable:
    """アルゴリズム関数をコンパイルして分布計算関数を返す"""
    return _default_engine.compile(algo_func)


# アルゴリズム記述用の便利クラス・関数
class AlgorithmBuilder:
    """アルゴリズム記述用のビルダークラス"""
    
    @staticmethod
    def vector_add(x_list: List[Dist], y_list: Union[List[Dist], Dist]) -> List[Dist]:
        """ベクトル加法"""
        if isinstance(y_list, Dist):
            # スカラーとベクトルの加法
            return [Add.apply(x, y_list) for x in x_list]
        else:
            # ベクトル同士の加法
            if len(x_list) != len(y_list):
                raise ValueError("Vector lengths must match")
            return [Add.apply(x, y) for x, y in zip(x_list, y_list)]
    
    @staticmethod
    def create_laplace_vector(b: float, size: int) -> List[Dist]:
        """ラプラス分布のベクトルを作成"""
        laplace = Laplace(b=b, size=size)
        return laplace.to_dist()

    @staticmethod
    def create_exponential_vector(b: float, size: int) -> List[Dist]:
        """指数分布のベクトルを作成"""
        exp = Exponential(b=b, size=size)
        return exp.to_dist()


# 使いやすさのためのエイリアス
def Laplace_dist(b: float, size: int = None) -> Union[Dist, List[Dist]]:
    """ラプラス分布を作成（アルゴリズム記述用）"""
    return create_laplace_noise(b=b, size=size)


def Exponential_dist(b: float, size: int = None) -> Union[Dist, List[Dist]]:
    """指数分布を作成（アルゴリズム記述用）"""
    return create_exponential_noise(b=b, size=size)


def vector_argmax(distributions: List[Dist]) -> Dist:
    """ベクトルのargmaxを計算（アルゴリズム記述用）"""
    return argmax_distribution(distributions)


def vector_max(distributions: List[Dist]) -> Dist:
    """ベクトルのmaxを計算（アルゴリズム記述用）"""
    return max_distribution(distributions)


def vector_min(distributions: List[Dist]) -> Dist:
    """ベクトルのminを計算（アルゴリズム記述用）"""
    return min_distribution(distributions)
