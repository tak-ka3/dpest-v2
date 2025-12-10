# dpest

差分プライバシー ε 推定フレームワーク

## 概要

`dpest` は差分プライバシー (DP) アルゴリズムを「確率分布オブジェクト (`Dist`) の計算グラフ」として表現し、その分布を解析またはサンプリングによって求めることで ε を推定するPythonライブラリです。各ステップはスカラー値ではなく分布を入力・出力とする演算で記述します。

**主な特徴**:
- 確率分布の混合表現（離散分布 + 連続分布）
- 依存関係を自動検出し、解析モード/サンプリングモードを自動選択
- 18種類の差分プライバシーアルゴリズムを実装

**実装されているアルゴリズム**:
- Sparse Vector Technique (SVT1-6, NumericalSVT, SVT34Parallel)
- Report Noisy Max (ReportNoisyMax1-4, NoisyMaxSum)
- RAPPOR (OneTimeRAPPOR, RAPPOR)
- Noisy Histogram (NoisyHist1-2)
- Laplace Mechanism (LaplaceMechanism, LaplaceParallel)
- その他 (PrefixSum, TruncatedGeometric)

## クイックスタート

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd dpest1

# 依存関係のインストール
pip install -r requirements.txt
```

### 基本的な使い方

```bash
# 特定のアルゴリズムのプライバシー損失を推定
python examples/privacy_loss_single_algorithm.py SVT1 \
  --config examples/privacy_loss_single_config.json

# 別のアルゴリズムを試す
python examples/privacy_loss_single_algorithm.py ReportNoisyMax1 \
  --config examples/privacy_loss_single_config.json

# 複数アルゴリズムの比較レポートを生成
python examples/privacy_loss_report.py

# 生成されたレポートを確認
cat docs/privacy_loss_report.md
```

### プライバシー損失の推定手法

dpestは2つの推定モードを提供します：

1. **解析モード（Analytic Mode）**: 入力の確率変数が互いに独立な場合
   - 格子近似（g=1000点）による数値計算
   - 高速・高精度（相対誤差 0-6%）

2. **サンプリングモード（Sampling Mode）**: 依存関係がある場合（Branch演算、共通変数参照など）
   - Monte Carloサンプリング（N=100,000サンプル）
   - 依存関係を正確に扱える（相対誤差 ±6%）

詳細は[PRIVACY_LOSS_ESTIMATION.md](docs/PRIVACY_LOSS_ESTIMATION.md)を参照してください。

## アーキテクチャ

### ディレクトリ構成

```
dpest1/
├── dpest/                      # メインパッケージ
│   ├── core.py                 # 確率分布クラス (Dist)
│   ├── noise.py                # ノイズ分布 (Laplace, Exponential)
│   ├── operations/             # 確率分布演算 (Add, Max, Argmax, Branch等)
│   ├── algorithms/             # DPアルゴリズム実装 (SVT, Noisy Max, RAPPOR等)
│   ├── analysis/               # プライバシー損失推定
│   └── utils/                  # ユーティリティ
├── examples/                   # 使用例
└── docs/                       # ドキュメント
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
        _joint_samples: サンプリングモード用のジョイントサンプル
    """
```

**特徴**:
- 離散分布と連続分布の混合表現
- FFTベースの効率的な畳み込み計算（解析モード）
- サンプリングベースの依存関係処理（サンプリングモード）

**主要メソッド**:
- `from_atoms()`: 離散分布の作成
- `from_density()`: 連続分布の作成
- `deterministic()`: 確定値（退化分布）の作成
- `sample()`: サンプリング
- `has_joint_samples()`: サンプリングモードかどうかの判定

#### 2. @auto_dist() デコレータ（algorithms/wrappers.py）

アルゴリズム関数を自動的にコンパイルして、プライバシー損失推定用の関数を提供します。

```python
@auto_dist()
def svt1(queries, eps: float, threshold: float, cutoff: int) -> list[Dist]:
    """SVT1アルゴリズム"""
    # アルゴリズムの実装
    ...
```

**機能**:
1. アルゴリズム関数を実行して出力分布を計算
2. 依存関係を自動検出（Branch演算など）
3. 解析モードまたはサンプリングモードを自動選択
4. `_dist_func`属性として分布計算関数を保存

#### 3. estimate_algorithm() 関数（analysis/estimation.py）

隣接データセットのペアに対してプライバシー損失εを推定します。

```python
def estimate_algorithm(
    name: str,
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    dist_func: Optional[Callable[..., Sequence[Dist] | Dist]] = None,
    eps: float = 0.1,
    n_samples: int = 100_000,
    extra: Optional[Iterable] = None,
) -> float | tuple[float, str]:
    """アルゴリズムのプライバシー損失を推定"""
    ...
```

**処理フロー**:
1. 各隣接データセットペアに対して出力分布を計算
2. プライバシー損失を計算（離散/連続/ジョイント分布に応じて）
3. 全ペアの最大値を返す

#### 4. 演算オペレーション（operations/）

dpestでは以下の確率分布演算を提供します。詳細は[OPERATIONS.md](OPERATIONS.md)を参照してください。

##### 基本演算

- **Add**: `Z = X + Y` - 2つの確率変数の和
- **Affine**: `Z = aX + b` - アフィン変換
- **Mul**: `Z = X × Y` - 2つの確率変数の積

##### 順序統計量

- **Max**: `Z = max(X₁, ..., Xₖ)` - 最大値
- **Min**: `Z = min(X₁, ..., Xₖ)` - 最小値
- **Argmax**: `Z = argmax(X₁, ..., Xₖ)` - 最大値のインデックス（離散分布）

##### 条件分岐

- **Compare/geq**: `C = 𝟙_{X ≥ Y}` - 比較演算（離散分布を返す）
- **Branch**: 条件分岐による混合分布

##### その他

- **Vector演算**: ベクトルに対する一括演算（add, argmax等）
- **PrefixSum**: 累積和演算

#### 5. ノイズ機構（noise.py）

- **Laplace**: ラプラス分布 `f(z) = (1/2b)exp(-|z|/b)`
- **Exponential**: 指数分布 `f(z) = (1/b)exp(-z/b)` (z ≥ 0)

格子近似を用いて効率的に計算します。

## 設計原則

dpestの設計原則と実装の詳細については、以下のドキュメントを参照してください：

- **実行モードの選択**: [PRIVACY_LOSS_ESTIMATION.md](docs/PRIVACY_LOSS_ESTIMATION.md) - 解析モードとサンプリングモードの違い
- **精度と性能**: [ACCURACY_AND_PERFORMANCE_ANALYSIS.md](docs/ACCURACY_AND_PERFORMANCE_ANALYSIS.md) - 各モードの精度保証と計算量
- **演算の詳細**: [OPERATION_DETAILS.md](docs/OPERATION_DETAILS.md) - 各演算の実装アルゴリズム
- **数学的背景**: [DESIGN.md](docs/DESIGN.md) - 確率分布演算の数学的定義


## ドキュメント

docs/配下には以下のドキュメントが含まれています：

### 主要ドキュメント

- **[DESIGN.md](docs/DESIGN.md)**: 各演算の数学的定義と実装方針
- **[OPERATIONS.md](docs/OPERATIONS.md)**: 演算の一覧と概要
- **[OPERATION_DETAILS.md](docs/OPERATION_DETAILS.md)**: 演算の詳細な計算方法とアルゴリズム
- **[ACCURACY_AND_PERFORMANCE_ANALYSIS.md](docs/ACCURACY_AND_PERFORMANCE_ANALYSIS.md)**: 解析手法とサンプリング手法の精度・性能分析
- **[PRIVACY_LOSS_ESTIMATION.md](docs/PRIVACY_LOSS_ESTIMATION.md)**: プライバシー損失の推定方法の詳細

### 計算量解析

- **[ARGMAX_COMPLEXITY.md](docs/ARGMAX_COMPLEXITY.md)**: Argmax演算の計算量詳細解析

## よくある質問

### Q: どこから始めればいい？

**A**: まず以下を実行してみてください：

```bash
# 複数アルゴリズムの比較レポートを生成
python examples/privacy_loss_report.py

# 生成されたレポートを確認
cat docs/privacy_loss_report.md
```

その後、[PRIVACY_LOSS_ESTIMATION.md](docs/PRIVACY_LOSS_ESTIMATION.md)でdpestの推定手法を理解してください。

### Q: 各ドキュメントの役割は？

**A**:
- **README.md**（このファイル）: リポジトリ全体の構造とクイックスタート
- **[PRIVACY_LOSS_ESTIMATION.md](docs/PRIVACY_LOSS_ESTIMATION.md)**: プライバシー損失の推定方法（最初に読むべき）
- **[ACCURACY_AND_PERFORMANCE_ANALYSIS.md](docs/ACCURACY_AND_PERFORMANCE_ANALYSIS.md)**: 精度と性能の詳細分析
- **[OPERATIONS.md](docs/OPERATIONS.md)**: 演算の一覧と使い方
- **[OPERATION_DETAILS.md](docs/OPERATION_DETAILS.md)**: 演算の実装詳細
- **[DESIGN.md](docs/DESIGN.md)**: 数学的定義と設計思想
- **[ARGMAX_COMPLEXITY.md](docs/ARGMAX_COMPLEXITY.md)**: Argmax演算の計算量解析

### Q: 実装されているアルゴリズムは？

**A**: 以下の18種類：
- Laplace Mechanism系: LaplaceMechanism, LaplaceParallel
- Noisy Histogram系: NoisyHist1, NoisyHist2
- Report Noisy Max系: ReportNoisyMax1-4, NoisyMaxSum
- RAPPOR系: OneTimeRAPPOR, RAPPOR
- SVT系: SVT1-6, SVT34Parallel, NumericalSVT
- その他: PrefixSum, TruncatedGeometric

各アルゴリズムの詳細は`dpest/algorithms/`配下のソースコードを参照してください。
