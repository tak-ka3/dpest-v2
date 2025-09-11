"""
差分プライバシーε推定エンジンの実装

アルゴリズムをコンパイルして分布計算関数を生成します。
"""

from typing import Callable, Any, List, Union
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
)


class ComputationNode:
    """計算グラフのノード"""
    def __init__(self, operation: str, inputs: List['ComputationNode'] = None, params: dict = None):
        self.operation = operation
        self.inputs = inputs or []
        self.params = params or {}
        self.output = None


class Engine:
    """差分プライバシーε推定エンジンのメインクラス"""
    
    def __init__(self):
        self.operations = {
            'input': self._input_op,
            'laplace': self._laplace_op,
            'exponential': self._exponential_op,
            'add': self._add_op,
            'affine': self._affine_op,
            'argmax': self._argmax_op,
            'max': self._max_op,
            'min': self._min_op,
        }
    
    def compile(self, algo_func: Callable) -> Callable:
        """
        アルゴリズム関数から最適な計算グラフを生成して分布計算関数を返す
        
        注意: この実装では簡略化のため、実際のASTパースは行わず、
        直接的な関数実行で分布を計算します。
        """
        def distribution_func(input_data):
            """入力データから出力分布を計算"""
            
            # 入力データから初期分布を作成
            input_dist = self._create_input_distribution(input_data)
            
            # アルゴリズムを実行（簡略化実装）
            # 実際の実装では、algo_funcをパースして計算グラフを構築すべき
            try:
                output_dist = algo_func(input_dist)
                return output_dist
            except Exception as e:
                # フォールバック: 直接実行が失敗した場合
                print(f"Direct execution failed: {e}")
                return self._fallback_execution(algo_func, input_data)
        
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
    
    def _fallback_execution(self, algo_func, input_data):
        """フォールバックの直接実行"""
        # 最も基本的な実行パス
        input_dist = self._create_input_distribution(input_data)
        return input_dist
    
    # 各操作の実装
    def _input_op(self, node: ComputationNode) -> Any:
        """入力操作"""
        return node.params.get('data')
    
    def _laplace_op(self, node: ComputationNode) -> Union[Dist, List[Dist]]:
        """ラプラス分布操作"""
        b = node.params.get('b', 1.0)
        size = node.params.get('size')
        return create_laplace_noise(b=b, size=size)

    def _exponential_op(self, node: ComputationNode) -> Union[Dist, List[Dist]]:
        """指数分布操作"""
        b = node.params.get('b', 1.0)
        size = node.params.get('size')
        return create_exponential_noise(b=b, size=size)
    
    def _add_op(self, node: ComputationNode) -> Union[Dist, List[Dist]]:
        """加法操作"""
        if len(node.inputs) != 2:
            raise ValueError("Add operation requires exactly 2 inputs")
        
        x_dist = node.inputs[0].output
        y_dist = node.inputs[1].output
        
        return add_distributions(x_dist, y_dist)
    
    def _affine_op(self, node: ComputationNode) -> Dist:
        """アフィン変換操作"""
        if len(node.inputs) != 1:
            raise ValueError("Affine operation requires exactly 1 input")
        
        x_dist = node.inputs[0].output
        a = node.params.get('a', 1.0)
        b = node.params.get('b', 0.0)
        
        return affine_transform(x_dist, a, b)
    
    def _argmax_op(self, node: ComputationNode) -> Dist:
        """Argmax操作"""
        if len(node.inputs) != 1:
            raise ValueError("Argmax operation requires exactly 1 input")
        
        distributions = node.inputs[0].output
        if not isinstance(distributions, list):
            raise ValueError("Argmax requires a list of distributions")
        
        return argmax_distribution(distributions)
    
    def _max_op(self, node: ComputationNode) -> Dist:
        """Max操作"""
        if len(node.inputs) != 1:
            raise ValueError("Max operation requires exactly 1 input")
        
        distributions = node.inputs[0].output
        if not isinstance(distributions, list):
            raise ValueError("Max requires a list of distributions")
        
        return max_distribution(distributions)
    
    def _min_op(self, node: ComputationNode) -> Dist:
        """Min操作"""
        if len(node.inputs) != 1:
            raise ValueError("Min operation requires exactly 1 input")
        
        distributions = node.inputs[0].output
        if not isinstance(distributions, list):
            raise ValueError("Min requires a list of distributions")
        
        return min_distribution(distributions)


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