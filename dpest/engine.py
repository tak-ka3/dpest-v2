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
    Max,
    Min,
    Argmax,
    PrefixSum,
    Compare,
    Condition,
    max_op,
    min_op,
    argmax,
    vector_add,
    vector_argmax,
    vector_max,
    vector_min,
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
            # リストの場合は各要素のノードを解析
            for r in result:
                if hasattr(r, 'node'):
                    self._analyze_node(r.node)

            # 個別要素がサンプリング必要かチェック
            needs_sampling_individually = any(getattr(r.node, 'needs_sampling', False)
                                              for r in result if hasattr(r, 'node'))

            # リスト要素間の依存関係をチェック（SVT5/SVT6のような共通依存性を検出）
            needs_sampling_cross_deps = False
            if not needs_sampling_individually:
                all_deps = [getattr(r, 'dependencies', set()) for r in result]
                # リストの任意の2要素間で依存関係の重なりをチェック
                for i in range(len(all_deps)):
                    for j in range(i + 1, len(all_deps)):
                        if all_deps[i] & all_deps[j]:
                            # 共通依存あり → サンプリングモード
                            needs_sampling_cross_deps = True
                            break
                    if needs_sampling_cross_deps:
                        break

            mode = 'sampling' if (needs_sampling_individually or needs_sampling_cross_deps) else 'analytic'
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
        ベクトル化により複数サンプルを同時生成して高速化します。
        """
        import numpy as np

        def sample_function_vectorized(n):
            """
            ベクトル化されたサンプリング: n個のサンプルを同時生成

            dpest_referenceの_calc_pdf_by_sampling_rec_vecを参考に実装。
            各Distオブジェクトから一度にn個のサンプルを生成することで、
            ループオーバーヘッドを削減する。
            """
            vectorized_cache = {}

            def realize_batch(value, n_samples):
                """
                確率変数からn_samples個のサンプルをベクトル化して生成

                戻り値:
                  - スカラー出力の場合: shape (n_samples,) の配列
                  - リスト出力の場合: shape (n_samples, len(list)) の配列
                """
                cache_key = id(value)

                if cache_key in vectorized_cache:
                    return vectorized_cache[cache_key]

                if isinstance(value, Dist):
                    # Distオブジェクトからベクトル化サンプリング
                    if hasattr(value, '_vectorized_sample'):
                        # カスタムベクトル化サンプリングがあれば使用
                        samples = value._vectorized_sample(n_samples)
                        vectorized_cache[cache_key] = samples
                        return samples

                    # デフォルト: _sample_funcを使用
                    if value._sample_func is not None:
                        # 各サンプルごとに独立したcacheで実行
                        samples = np.empty(n_samples)
                        for i in range(n_samples):
                            samples[i] = value._sample({})
                        vectorized_cache[cache_key] = samples
                        return samples

                    # サンプラーがある場合
                    if value.sampler is not None:
                        samples = value.sample(n_samples)
                        samples = np.asarray(samples)
                        if samples.ndim > 1:
                            idx = value.sampler_index or 0
                            samples = samples[:, idx]
                        vectorized_cache[cache_key] = samples
                        return samples

                    # 離散分布の場合
                    if value.atoms:
                        values, weights = zip(*value.atoms)
                        weights = np.asarray(weights, dtype=float)
                        total = weights.sum()
                        if total > 0:
                            weights = weights / total
                            samples = np.random.choice(values, size=n_samples, p=weights)
                        else:
                            samples = np.full(n_samples, values[0])
                        vectorized_cache[cache_key] = samples
                        return samples

                    # 連続密度の場合
                    if value.density:
                        x = value.density.get('x')
                        f = value.density.get('f')
                        dx = value.density.get('dx', 1.0)
                        probs = np.asarray(f, dtype=float) * float(dx)
                        probs = np.clip(probs, 0.0, None)
                        total = probs.sum()
                        if total > 0:
                            probs = probs / total
                            samples = np.random.choice(x, size=n_samples, p=probs)
                        else:
                            samples = np.full(n_samples, x[0])
                        vectorized_cache[cache_key] = samples
                        return samples

                    # フォールバック
                    samples = np.zeros(n_samples)
                    vectorized_cache[cache_key] = samples
                    return samples

                if isinstance(value, list):
                    if not value:
                        empty = np.empty((n_samples, 0))
                        vectorized_cache[cache_key] = empty
                        return empty
                    # リストの各要素をサンプリング
                    # 結果は (n_samples, len(value)) の配列

                    # すべての要素が_sample_funcを持つか確認
                    all_have_sample_func = all(
                        isinstance(v, Dist) and v._sample_func is not None
                        for v in value
                    )

                    if all_have_sample_func:
                        # SVT1のような状態を持つアルゴリズム:
                        # 各サンプルごとにすべての要素が同じcacheを共有する必要がある
                        samples = np.empty((n_samples, len(value)))
                        for i in range(n_samples):
                            cache = {}  # 各サンプルごとに新しいcache
                            for j, v in enumerate(value):
                                samples[i, j] = v._sample(cache)
                        vectorized_cache[cache_key] = samples
                        return samples
                    else:
                        # 独立な分布: 各要素を並列にベクトル化サンプリング
                        element_samples = []
                        for v in value:
                            elem_batch = realize_batch(v, n_samples)
                            element_samples.append(elem_batch)
                        stacked = np.column_stack([
                            eb if eb.ndim > 1 else eb.reshape(n_samples, 1)
                            for eb in element_samples
                        ])
                        vectorized_cache[cache_key] = stacked
                        return stacked

                if isinstance(value, tuple):
                    if len(value) == 0:
                        empty = np.empty((n_samples, 0))
                        vectorized_cache[cache_key] = empty
                        return empty
                    tuple_list = list(value)
                    element_batches = [realize_batch(v, n_samples) for v in tuple_list]
                    stacked = np.column_stack([
                        eb if eb.ndim > 1 else eb.reshape(n_samples, 1)
                        for eb in element_batches
                    ])
                    vectorized_cache[cache_key] = stacked
                    return stacked

                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        arr = np.full(n_samples, value.item())
                    elif value.ndim == 1:
                        arr = np.tile(value, (n_samples, 1))
                    else:
                        expanded = np.repeat(value[None, ...], n_samples, axis=0)
                        arr = expanded.reshape(n_samples, -1)
                    vectorized_cache[cache_key] = arr
                    return arr

                if isinstance(value, (bool, np.bool_)):
                    arr = np.full(n_samples, bool(value), dtype=np.bool_)
                    vectorized_cache[cache_key] = arr
                    return arr

                if isinstance(value, (int, np.integer)):
                    arr = np.full(n_samples, int(value), dtype=np.int64)
                    vectorized_cache[cache_key] = arr
                    return arr

                if isinstance(value, (float, np.floating)):
                    arr = np.full(n_samples, float(value))
                    vectorized_cache[cache_key] = arr
                    return arr

                if isinstance(value, str):
                    arr = np.full(n_samples, value, dtype=object)
                    vectorized_cache[cache_key] = arr
                    return arr

                # スカラー値（その他の型）は object 配列として扱う
                arr = np.full(n_samples, value, dtype=object)
                vectorized_cache[cache_key] = arr
                return arr

            # アルゴリズムを一度実行して構造を取得
            result_template = algo_func(input_dist)

            # ベクトル化サンプリングを実行
            try:
                samples = realize_batch(result_template, n)
                return samples
            except Exception as e:
                # ベクトル化失敗時は従来の逐次サンプリングにフォールバック
                print(f"Vectorized sampling failed: {e}")
                print("Falling back to sequential sampling...")
                return sample_function_sequential(n)

        def sample_function_sequential(n):
            """従来の逐次サンプリング（フォールバック用）"""
            def realize(value, cache):
                if isinstance(value, Dist):
                    return value._sample(cache)
                if isinstance(value, list):
                    return [realize(v, cache) for v in value]
                if isinstance(value, tuple):
                    return [realize(v, cache) for v in value]
                if isinstance(value, np.ndarray):
                    return np.array(value, copy=True)
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                if isinstance(value, (int, np.integer)):
                    return int(value)
                if isinstance(value, (float, np.floating)):
                    return float(value)
                return value

            # 最初のサンプルで形状を取得
            cache = {}
            result = algo_func(input_dist)
            first_sample = realize(result, cache)

            # 配列の形状を決定
            if isinstance(first_sample, list):
                sample_shape = (n, len(first_sample))
                samples = np.empty(sample_shape, dtype=float)
                samples[0] = np.asarray(first_sample, dtype=float)
                start_idx = 1
            elif isinstance(first_sample, np.ndarray):
                if first_sample.ndim == 1:
                    sample_shape = (n, first_sample.shape[0])
                    samples = np.empty(sample_shape, dtype=first_sample.dtype)
                    samples[0] = first_sample
                else:
                    sample_shape = (n, first_sample.size)
                    samples = np.empty(sample_shape, dtype=first_sample.dtype)
                    samples[0] = first_sample.reshape(-1)
                start_idx = 1
            else:
                samples = np.empty(n, dtype=type(first_sample) if isinstance(first_sample, (bool, np.bool_)) else float)
                samples[0] = first_sample
                start_idx = 1

            for i in range(start_idx, n):
                cache = {}
                result = algo_func(input_dist)
                sample_val = realize(result, cache)
                if isinstance(sample_val, list):
                    samples[i] = np.asarray(sample_val, dtype=float)
                elif isinstance(sample_val, np.ndarray):
                    if sample_val.ndim == 1 and samples.ndim == 2:
                        samples[i] = sample_val
                    elif samples.ndim == 2:
                        samples[i] = sample_val.reshape(-1)
                    else:
                        samples[i] = sample_val
                else:
                    samples[i] = sample_val

            return samples

        if sampler is not None and raw_input is not None:
            sample_array = np.asarray(sampler(raw_input, n_samples))
        else:
            # ベクトル化サンプリングを試行
            sample_array = sample_function_vectorized(n_samples)

        # サンプル配列から軽量なDistオブジェクトを構築
        # Note: サンプリングモードでは _joint_samples のみがε計算に使われるため、
        # atoms/density は最小限のダミーで良い
        if sample_array.ndim == 1:
            # 1次元の場合: ダミーのDistを1つ作成
            result = Dist.from_atoms([(0.0, 1.0)])  # ダミーatom
            result._joint_samples = sample_array.reshape(-1, 1)
            result._joint_samples_column = 0
        else:
            # 多次元の場合: 各列に対してダミーのDistを作成
            result = []
            for i in range(sample_array.shape[1]):
                dist = Dist.from_atoms([(0.0, 1.0)])  # ダミーatom
                dist._joint_samples = sample_array
                dist._joint_samples_column = i
                result.append(dist)

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


# vector_add, vector_argmax, vector_max, vector_min are now imported from operations module
# These were previously defined here but have been moved to operations.vector_ops for better organization
