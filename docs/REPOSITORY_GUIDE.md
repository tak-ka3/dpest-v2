# dpest リポジトリ解説

## 概要

`dpest` は差分プライバシーのε（イプシロン）推定を行うPythonライブラリです。確率分布の計算と変換を通じて、差分プライバシーアルゴリズムのプライバシーパラメータを推定します。

## プロジェクトの目的

差分プライバシーアルゴリズムのプライバシー損失（ε）を正確に推定することで、理論値と実際のプライバシー保証のギャップを検証します。特に以下のような機構の解析が可能です：

- Sparse Vector Technique (SVT)
- Report Noisy Max
- RAPPOR
- Parallel Composition
- Prefix Sum

## アーキテクチャ

### ディレクトリ構成

```
dpest1/
├── dpest/                      # メインパッケージ
│   ├── core.py                 # 中核クラス（Dist, Interval, Node）
│   ├── engine.py               # コンパイルエンジン
│   ├── noise.py                # ノイズ機構（Laplace, Exponential）
│   ├── operations/             # 確率分布演算
│   │   ├── operations.py       # 基本演算（Add, Affine）
│   │   ├── max_op.py          # Max/Min 演算
│   │   ├── argmax_op.py       # Argmax 演算
│   │   ├── condition_op.py    # 条件付き分岐
│   │   ├── prefix_sum_op.py   # 累積和
│   │   ├── sample_op.py       # サンプリング演算
│   │   └── geometric_op.py    # 幾何分布演算
│   ├── mechanisms/             # 差分プライバシー機構
│   │   ├── laplace.py         # ラプラス機構
│   │   ├── geometric.py       # 幾何機構
│   │   ├── sparse_vector_technique.py  # SVT実装
│   │   ├── report_noisy_max.py         # Report Noisy Max
│   │   ├── rappor.py          # RAPPOR
│   │   ├── parallel.py        # 並列合成
│   │   ├── prefix_sum.py      # Prefix Sum
│   │   └── noisy_hist.py      # Noisy Histogram
│   └── utils/                  # ユーティリティ
│       ├── privacy.py          # プライバシー損失計算
│       ├── input_patterns.py  # 入力パターン生成
│       └── zero.py             # ゼロ処理
├── examples/                   # 使用例
│   ├── example.py             # 基本的な使用例
│   ├── privacy_loss_single_algorithm.py  # 単一アルゴリズム解析
│   ├── privacy_loss_report.py            # 複数アルゴリズム比較
│   └── compare_svt1_methods.py           # SVT1手法比較
├── tests/                      # テストスイート
│   ├── test_operations.py     # 演算テスト
│   ├── test_svt*_joint.py     # SVT変種のテスト
│   └── test_privacy_loss.py   # プライバシー損失テスト
└── docs/                       # ドキュメント
    ├── DESIGN.md              # 設計仕様
    ├── OPERATION.md           # 演算の計算方法
    ├── DEPENDENCY_DESIGN.md   # 依存関係設計
    ├── DEPENDENCY_ANALYSIS.md # 依存関係解析
    ├── SVT*.md                # SVT関連ドキュメント
    └── privacy_loss_report.md # プライバシー損失レポート
```

### 中核コンポーネント

#### 1. Dist クラス（core.py）

確率分布を表現する中心的なクラスです。

```python
class Dist:
    """確率分布の表現

    Attributes:
        atoms: 点質量（離散分布）のリスト [(value, weight), ...]
        density: 連続密度の格子近似 {'x': grid_x, 'f': grid_f, 'dx': dx}
        support: サポート区間のリスト
        error_bounds: 誤差上界の情報
        dependencies: 依存関係の識別子集合
    """
```

**特徴**:
- 離散分布と連続分布の混合表現
- FFTベースの効率的な畳み込み計算
- 誤差管理（切り捨て誤差、補間誤差、数値積分誤差）
- 依存関係の追跡

**主要メソッド**:
- `from_atoms()`: 離散分布の作成
- `from_density()`: 連続分布の作成
- `deterministic()`: 確定値（退化分布）の作成
- `sample()`: サンプリング

#### 2. Engine クラス（engine.py）

アルゴリズムをコンパイルして実行可能な分布計算関数を生成します。

```python
class Engine:
    def compile(self, algo_func: Callable) -> Callable:
        """アルゴリズム関数をコンパイル"""
        # 1. 計算グラフの構築
        graph = self._build_computation_graph(algo_func)
        # 2. グラフの最適化
        optimized = self._optimize_graph(graph)
        # 3. 実行計画の作成
        return distribution_func
```

**処理フロー**:
1. アルゴリズム関数から計算グラフを構築
2. 計算グラフを最適化
3. 依存関係を解析し実行モード（解析的 or サンプリング）を決定
4. 分布計算関数を返す

#### 3. 演算オペレーション（operations/）

##### 基本演算

- **Affine**: `Z = aX + b` - 線形変換
  - 連続部: ヤコビアン補正による変数変換
  - 点質量: 座標変換のみ

- **Add**: `Z = X + Y` - 独立な確率変数の和
  - 連続+連続: FFT畳み込み
  - 離散+離散: 組合せ列挙
  - 離散+連続: シフトと重ね合わせ

##### 順序統計量

- **Max**: `Z = max(X₁, ..., Xₖ)`
  - 累積分布関数: `F_max(z) = ∏ᵢ Fᵢ(z)`
  - 確率密度関数: `f_max(z) = Σᵢ fᵢ(z) ∏ⱼ≠ᵢ Fⱼ(z)`

- **Min**: `Z = min(X₁, ..., Xₖ)`
  - 公式: `1 - F_min(z) = ∏ᵢ (1 - Fᵢ(z))`

##### 離散出力

- **Argmax**: インデックス分布
  - `P(argmax = i) = ∫ fᵢ(x) ∏ⱼ≠ᵢ Fⱼ(x) dx`
  - 格子上で数値積分

##### 条件分岐

- **Compare**: 確率変数の大小比較
- **Condition**: 条件付き混合分布

#### 4. ノイズ機構（noise.py）

- **Laplace**: ラプラス分布 `f(z) = (1/2b)exp(-|z|/b)`
- **Exponential**: 指数分布 `f(z) = (1/b)exp(-z/b)` (z ≥ 0)

格子近似とFFTを用いて効率的に計算します。

## 設計原則

### 混合分布の処理

- 連続部分: FFTベースの数値計算
- 点質量（atoms）: 個別に処理後に合成
- 両者を統一的に扱い、最終的に混合分布として表現

### 誤差管理

```
total_error = err_trunc + err_interp + err_quad
```

- `err_trunc`: 格子範囲外の切り捨て誤差
- `err_interp`: 補間誤差
- `err_quad`: 数値積分誤差

### 依存関係の扱い

- **独立**: 個別分布の畳み込み・積で計算可能
- **依存あり**: サンプリングベースの計算にフォールバック

依存関係は`dependencies`集合で追跡され、重なりがある場合は自動的にサンプリングモードに切り替わります。

### 格子近似

連続分布は格子上で離散化して計算：
- 一様格子 `x = [x₀, x₁, ..., xₙ]`
- 格子幅 `dx`
- 密度値 `f = [f₀, f₁, ..., fₙ]`

## 使用例

### 基本的な使い方

```python
from dpest import compile, Laplace, Argmax, Add

# アルゴリズムを定義
def noisy_argmax(x):
    """ラプラスノイズを加えたargmax"""
    noises = Laplace(b=1.0, size=len(x)).to_dist()
    noisy_x = [Add.apply(xi, ni) for xi, ni in zip(x, noises)]
    return Argmax.apply(noisy_x)

# コンパイル
algo_compiled = compile(noisy_argmax)

# 実行
D = [1.0, 2.0, 3.0, 4.0, 5.0]
output_dist = algo_compiled(D)

# 結果の確認
print(f"Atoms (discrete probabilities): {output_dist.atoms}")
# 例: [(0, 0.01), (1, 0.05), (2, 0.14), (3, 0.30), (4, 0.50)]
```

### プライバシー損失の推定

```python
from dpest.utils.privacy import estimate_epsilon

# 隣接データセット
D       = [1, 1, 1, 1, 1]
D_prime = [1, 1, 1, 1, 0]

# 両方の出力分布を計算
P = algo_compiled(D)
Q = algo_compiled(D_prime)

# ε推定
epsilon = estimate_epsilon(P, Q)
print(f"Estimated ε: {epsilon}")
```

### SVT（Sparse Vector Technique）の例

```python
from dpest.mechanisms.sparse_vector_technique import svt1_algorithm

# SVT1の定義（閾値判定タイプ）
c = 2  # カットオフ回数
T = 5.0  # 閾値
epsilon = 1.0

def my_svt1(queries):
    return svt1_algorithm(queries, T=T, c=c, epsilon=epsilon)

# コンパイルと実行
algo = compile(my_svt1)
queries = [3.0, 4.0, 6.0, 7.0, 5.0]
result_dist = algo(queries)
```

## テスト

### テストの実行

```bash
# すべてのテストを実行
pytest

# 特定のテストファイルを実行
pytest tests/test_operations.py

# 詳細な出力
pytest -v
```

### テストの種類

- `test_operations.py`: 基本演算のテスト
- `test_svt*_joint.py`: SVT各変種のテスト
- `test_privacy_loss.py`: プライバシー損失推定のテスト
- `test_geometric_op.py`: 幾何分布演算のテスト

### GitHub Actions

プッシュやプルリクエスト時に自動でテストが実行されます。また、Actionsタブから手動でも実行可能です。

## 例の実行

### 基本例

```bash
python examples/example.py
```

### 単一アルゴリズムの解析

```bash
python examples/privacy_loss_single_algorithm.py
```

### 複数アルゴリズムの比較レポート

```bash
python examples/privacy_loss_report.py
```

出力は `docs/privacy_loss_report.md` に保存されます。

### SVT手法の比較

```bash
python examples/compare_svt1_methods.py
```

## ドキュメント

### 設計ドキュメント

- **DESIGN.md**: 各演算の数学的定義と実装方針
- **OPERATION.md**: 演算の具体的な計算方法
- **DEPENDENCY_DESIGN.md**: 依存関係の設計
- **DEPENDENCY_ANALYSIS.md**: 依存関係の詳細解析

### アルゴリズム固有のドキュメント

- **SVT1_DISTRIBUTION.md**: SVT1の分布計算
- **SVT2_DISTRIBUTION.md**: SVT2の分布計算
- **SVT_ANALYTIC_DISTRIBUTION.md**: SVT解析的分布

## 開発ガイド

### 環境構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd dpest1

# 依存関係のインストール
pip install -r requirements.txt
```

### 依存パッケージ

- `numpy`: 数値計算
- `scipy`: 科学計算（FFT等）
- `pytest`: テストフレームワーク

### 新しい演算の追加

1. `dpest/operations/` に新しいファイルを作成
2. 演算クラスと計算関数を実装
3. `dpest/operations/__init__.py` にエクスポート追加
4. `dpest/engine.py` の `operations` 辞書に登録
5. テストを `tests/` に追加

### 新しい機構の追加

1. `dpest/mechanisms/` に新しいファイルを作成
2. 機構のアルゴリズムを実装
3. `dpest/mechanisms/__init__.py` にエクスポート追加
4. テストとサンプルコードを追加

## 数学的背景

### 確率分布の変換

差分プライバシー機構の出力分布を計算するため、確率変数の様々な変換を正確に計算します：

- **線形変換**: `Z = aX + b`
- **和**: `Z = X + Y` （畳み込み）
- **最大値・最小値**: `Z = max(X, Y)`, `Z = min(X, Y)`
- **Argmax**: `Z = argmax(X₁, ..., Xₖ)`

### プライバシー損失

2つの出力分布 P（データセット D）と Q（隣接データセット D'）に対して：

```
ε = max_output log(P(output) / Q(output))
```

これを数値的に計算してプライバシーパラメータεを推定します。

## 参考文献

差分プライバシーと本ライブラリで実装されている機構については、以下を参照：

- Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy.
- Lyu, M., et al. (2017). Understanding the sparse vector technique for differential privacy.

## ライセンスと貢献

プロジェクトの詳細については README.md を参照してください。

---

## よくある質問

### Q: サンプリングモードと解析モードの違いは？

**A**: 独立な確率変数のみを扱う場合は解析的に分布を計算できます（高速・正確）。依存関係がある場合は、サンプリングベースの近似計算にフォールバックします。

### Q: 誤差はどの程度？

**A**: 格子幅やサポート範囲の設定によりますが、通常は相対誤差1%以下に抑えられます。誤差情報は `Dist.error_bounds` に記録されます。

### Q: 新しいノイズ分布を追加するには？

**A**: `dpest/noise.py` を参考に、格子近似したPDFを返す関数を実装し、`Dist.from_density()` で分布オブジェクトを作成します。

### Q: 大規模な計算には向いている？

**A**: FFTベースの実装により、比較的効率的ですが、多次元分布や非常に細かい格子が必要な場合はメモリと計算時間が増大します。そのような場合はサンプリングベースの方が適している可能性があります。
