# dpest

差分プライバシー ε 推定フレームワーク

## 手法の概要

dpest は差分プライバシー (DP) アルゴリズムを「確率分布オブジェクト (`Dist`) の計算グラフ」として表現し、その分布を解析またはサンプリングによって求めることで ε を推定します。各ステップはスカラー値ではなく分布を入力・出力とする演算で記述します。

- **Atoms**: 離散的な確率質量（確定値など）を保持。
- **Density grids**: 連続分布を格子状に離散化。
- **Samplers**: 解析が難しい場合のカスタム乱数生成器。

`add`, `affine`, `geq`, `mux`, `max` などの演算はすべて `Dist` 上で定義されており、アルゴリズムを宣言的に組み立てられるようになっています。

### コンパイル時の解析とサンプリング切り替え

`dpest.engine.compile` によりアルゴリズムをコンパイルすると、エンジンは計算グラフを解析します。

1. **依存関係解析**:  同じ乱数源に依存する分布が存在すると、独立性の仮定が崩れるため解析モードでは扱えません。
2. **モード選択**:
   - 依存が問題にならなければ、解析的または格子上での数値計算により正確な分布を求めます。
   - 依存がある場合はサンプリングモードに切り替え、ベクトル化された Monte Carlo 実行で ` _joint_samples` を生成します。このジョイントサンプルは出力要素間の相関を保持します。

### プライバシー損失 (ε) 推定フロー

`dpest.analysis.estimate_algorithm` が ε 推定の共通ルートです。

1. 隣接データセットの組 (change-one adjacency) を生成。
2. 各組について、アルゴリズムの分布関数 (`svt1_dist`, `noisy_hist1_dist` など) を実行し、2つの出力分布を得る。
3. ε を計算:
   - ジョイントサンプルがあれば `epsilon_from_list_joint` で多次元ヒストグラム（NaN パターンも含む）を評価。
   - 厳密な分布オブジェクトなら `epsilon_from_dist` で解析的に計算。
4. すべての隣接ペアで最大の ε を採用。

こうすることで、分布演算で記述されたアルゴリズムはすべて同じ推定パイプラインに乗ります。

#### compile() 呼び出しの流れ

```python
from dpest.engine import compile
from dpest.algorithms.svt1 import svt1

algo = compile(lambda q: svt1(q, eps=0.1))
output_dist = algo(input_distributions)
```

- **1. 計算グラフ構築**: `compile()` はラムダを一度走らせ、内部で生成された `Dist` それぞれに `Node` 情報を付与して依存関係を収集します。
- **2. 依存解析**: `Node.dependencies` が重なる場合や `needs_sampling` フラグが立つ場合は解析的には扱えないため、エンジンはモードを `sampling` に設定します。
- **3. 実行プラン生成**: `Engine.ExecutionPlan` に解析結果と初期出力を格納。解析モードなら即結果を返す準備が整い、サンプリングモードなら `_execute_sampling` 用の設定が保存されます。
- **4. 実行**: `algo(input_distributions)` を呼ぶと `_create_input_distribution` が入力を `Dist` 化し、上記プランに基づいて解析またはサンプリングで分布を返します。サンプリングではベクトル化された Monte Carlo 実行で `Sampled.from_samples` を呼び、得られた `_joint_samples` が ε 推定の joint histogram に利用されます。

### `@auto_dist` によるラッパー自動生成

アルゴリズム実装（例: `svt1.py`, `noisy_hist1.py`）は `List[Dist]` を引数に取りますが、テストや examples では数値配列を渡したい場面が多いです。`@auto_dist` デコレータを使うと、次の処理を行う `*_dist` ラッパーが自動生成されます。

- 数値配列を確定値 `Dist` に変換。
- エンジンでコンパイルし、解析またはサンプリングを実行。
- `svt1_dist` などの名前で公開（手動の重複実装は不要）。

これにより SVT 系もノイズ機構系も同じ呼び出しフローを共有します。

## リポジトリ構成

- `dpest/core.py`: `Dist` や `Interval`、検証ロジック。
- `dpest/operations/`: 加算、比較、argmax/max、mux などの演算。
- `dpest/engine.py`: コンパイル・依存解析・解析/サンプリングモード切り替え。
- `dpest/analysis/`: 隣接ペア生成、ε 推定ヘルパー。
- `dpest/algorithms/`: アルゴリズムごとのモジュール（SVT, Noisy Hist, Laplace, RAPPOR 等）。すべて `@auto_dist` を使用。
- `examples/`: `privacy_loss_single_algorithm.py` などの実行スクリプト。
- `tests/`: 各演算・アルゴリズムを検証する pytest スイート。

## テストの実行

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest
```

## 実行例

SVT1 の ε を推定する例:

```bash
PYTHONPATH=. python examples/privacy_loss_single_algorithm.py SVT1 \
  --config examples/privacy_loss_single_config.json
```
