"""
差分プライバシーε推定エンジンの実装

アルゴリズムをコンパイルして分布計算関数を生成します。
"""

from dataclasses import dataclass
from typing import Callable, Any, List, Union, Optional
import numpy as np

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


@dataclass
class FallbackResult:
    value: Any
    sampler: Optional[Callable[[Any, int], np.ndarray]] = None


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
        # サンプリングモードで使用するサンプル数
        self.default_sampling_samples = 100

    def set_default_sampling_samples(self, n_samples: int):
        """サンプリングモードで使用するサンプル数を設定"""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self.default_sampling_samples = n_samples

    # アルゴリズムの関数から計算グラフを作る
    def _build_computation_graph(self, algo_func: Callable):
        """与えられたアルゴリズム関数から計算グラフを生成する"""

        if not callable(algo_func):
            raise TypeError("Algorithm must be callable")

        # 現状ではアルゴリズム関数そのものを計算グラフとして扱う
        return algo_func

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

        # ノード自身が既にneeds_samplingを持っている場合は尊重
        if getattr(node, 'needs_sampling', False):
            return True

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

    def _plan_execution(self, computation_graph: Callable, input_dist, raw_input) -> "Engine.ExecutionPlan":
        """計算グラフから出力分布の計算方法を決定する"""

        # 一度実行して計算グラフを構築し、依存関係を解析
        result = computation_graph(input_dist)
        sampler = None
        if isinstance(result, FallbackResult):
            sampler = result.sampler
            result = result.value

        # 結果のノードを解析してサンプリングが必要か判定
        if hasattr(result, 'node'):
            self._analyze_node(result.node)
            mode = 'sampling' if getattr(result.node, 'needs_sampling', False) else 'analytic'
        elif isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'node'):
            # リストの場合は最初の要素をチェック（簡易実装）
            for r in result:
                if hasattr(r, 'node'):
                    self._analyze_node(r.node)
            # いずれかがサンプリング必要なら全体もサンプリング
            mode = 'sampling' if any(getattr(r.node, 'needs_sampling', False)
                                     for r in result if hasattr(r, 'node')) else 'analytic'
        else:
            mode = 'analytic'

        return self.ExecutionPlan(
            mode=mode,
            graph=computation_graph,
            options={'result': result, 'sampler': sampler, 'raw_input': raw_input},
        )

    def _execute_algorithm(self, plan: "Engine.ExecutionPlan", input_dist):
        """実行計画に従ってアルゴリズムを実行し分布を得る"""

        if plan.mode == 'sampling':
            # サンプリングモード: アルゴリズムをサンプリングベースで再実行
            import warnings
            warnings.warn("Sampling mode detected due to variable dependencies. Executing with Monte Carlo sampling.",
                          UserWarning)
            return self._execute_sampling(
                plan.graph,
                input_dist,
                n_samples=self.default_sampling_samples,
                sampler=(plan.options or {}).get('sampler') if plan.options else None,
                raw_input=(plan.options or {}).get('raw_input') if plan.options else None,
            )
        else:
            # 解析モード: 既に計算済みの結果を返す
            return plan.options.get('result')

    def _execute_sampling(self, algo_func: Callable, input_dist, n_samples: int = 1000,
                          sampler: Optional[Callable[[Any, int], np.ndarray]] = None,
                          raw_input: Any = None):
        """サンプリングベースでアルゴリズムを実行

        アルゴリズムをn_samples回実行して、結果から分布を構築します。
        各実行では、すべての確率変数が独立にサンプリングされます。
        """
        import numpy as np
        from .operations import Sampled

        def sample_function(n):
            """アルゴリズムをn回実行してサンプルを生成"""
            samples = []
            for _ in range(n):
                # アルゴリズムを1回実行
                # 各確率変数は内部的にサンプリングされる
                result = algo_func(input_dist)

                # 結果を数値に変換
                def sample_from_dist(d):
                    """Distから1つのサンプルを取得"""
                    if d.atoms:
                        # atomsから確率的に選択
                        values = [v for v, _ in d.atoms]
                        weights = [w for _, w in d.atoms]
                        # 重みを正規化
                        total = sum(weights)
                        if total > 0:
                            weights = [w/total for w in weights]
                        return np.random.choice(values, p=weights)
                    elif hasattr(d, 'sampler') and d.sampler is not None:
                        # サンプラーを使用
                        try:
                            sample = d.sample(1)
                            if hasattr(sample, '__len__'):
                                return sample[0]
                            return sample
                        except:
                            return 0.0
                    else:
                        return 0.0

                if isinstance(result, Dist):
                    sample_val = sample_from_dist(result)
                elif isinstance(result, list):
                    # リストの場合は各要素をサンプリング
                    sample_val = []
                    for r in result:
                        if isinstance(r, Dist):
                            sample_val.append(sample_from_dist(r))
                        else:
                            sample_val.append(float(r))
                else:
                    sample_val = float(result)

                samples.append(sample_val)

            return np.array(samples)

        if sampler is not None and raw_input is not None:
            sample_array = np.asarray(sampler(raw_input, n_samples))
        else:
            sample_array = sample_function(n_samples)

        # 既存のサンプル配列から分布を構築（重複実行を回避）
        result = Sampled.from_samples(sample_array, bins=100)

        # 結合分布用にサンプル配列も保存
        if isinstance(result, list):
            for dist in result:
                dist._joint_samples = sample_array
        elif isinstance(result, Dist):
            result._joint_samples = sample_array

        return result

    def compile(self, algo_func: Callable) -> Callable:
        """アルゴリズム関数から計算グラフを生成して分布計算関数を返す（コンパイル）"""

        computation_graph = self._build_computation_graph(algo_func)

        def distribution_func(input_data):
            """入力データから出力分布を計算"""

            # 入力データから初期分布を作成
            input_dist = self._create_input_distribution(input_data)

            # 実行計画の作成とアルゴリズム実行
            plan = self._plan_execution(computation_graph, input_dist, input_data)
            output_dist = self._execute_algorithm(plan, input_dist)

            return output_dist

        return distribution_func

    def _create_input_distribution(self, input_data) -> Union[Dist, List[Dist]]:
        """入力データから初期分布を作成"""
        import numpy as np

        # numpy配列の場合はリストに変換
        if isinstance(input_data, np.ndarray):
            input_data = input_data.tolist()

        if isinstance(input_data, (list, tuple)):
            if all(isinstance(x, (int, float, np.integer, np.floating)) for x in input_data):
                # 数値リストの場合、各要素を確定値分布として作成
                return [Dist.deterministic(float(x)) for x in input_data]
            else:
                # 既にDistオブジェクトのリストの場合
                return input_data
        elif isinstance(input_data, (int, float, np.integer, np.floating)):
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


def set_sampling_samples(n_samples: int):
    """グローバルエンジンのサンプリング回数を設定"""
    _default_engine.set_default_sampling_samples(n_samples)


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
