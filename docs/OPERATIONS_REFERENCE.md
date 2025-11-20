# dpest オペレーション機能一覧表

## 1. 基本演算（Arithmetic Operations）

| オペレーション | 関数 | クラス | 入力 | 出力 | 機能説明 |
|--------------|------|--------|------|------|---------|
| **加算** | `add(x, y)` | `Add` | 2つの分布 | 分布 | Z = X + Y の分布を計算（FFTベース畳み込み） |
| **アフィン変換** | `affine(x, a, b)` | `Affine` | 分布、係数a、定数b | 分布 | Z = aX + b の分布を計算（ヤコビアン考慮） |

**使用例**:
```python
# ラプラスノイズの追加
noisy = add(value, Laplace(b=1.0).to_dist())

# スケーリングとシフト
scaled = affine(x, a=2.0, b=10.0)  # Z = 2X + 10
```

---

## 2. 比較・条件演算（Comparison & Conditional Operations）

| オペレーション | 関数 | クラス | 入力 | 出力 | 機能説明 |
|--------------|------|--------|------|------|---------|
| **大小比較** | `geq(x, y)` | `Compare` | 2つの分布/定数 | {0,1}分布 | X ≥ Y の指示値分布を返す |
| **条件分岐** | `branch(c, t, f)` | `Branch` | 条件分布、真値、偽値 | 混合分布 | 条件に基づいて値を選択（依存関係自動検出） |

> **注意**: 以前の `mux` および `condition` 関数は `branch` に統一されました。

### `branch(cond_dist, true_val, false_val)` - 統一条件分岐演算

**目的**: 条件に基づいて2つの値から1つを**選択**する（定数も可能、依存関係自動検出）

**動作原理**:
- 条件分布`cond_dist`が真（値≥0.5）となる確率を`p`とする
- 結果は`p × true_val + (1-p) × false_val`の混合分布
- 定数値を自動的に`Dist.deterministic()`に変換
- **共通の依存関係を自動検出**し、`needs_sampling`フラグを設定
- サンプリングモードへの自動フォールバック対応

**数学的表現**:
```
P(Z ∈ A) = P(条件=真) × P(true_val ∈ A) + P(条件=偽) × P(false_val ∈ A)
```

**使用場面**:
- 条件が確率的に決まる場合
- センチネル値（`NAN`, `PAD`）を扱う場合
- 同じ確率変数を複数回参照する場合（依存関係自動処理）

**例**:
```python
from dpest.operations import branch, geq

# 閾値を超える確率が70%の場合
cond = geq(noisy_value, threshold)  # P(cond=1) = 0.7, P(cond=0) = 0.3

result = branch(cond, 100.0, 10.0)
# 結果: 70%の確率で100、30%の確率で10の混合分布
# atoms = [(100.0, 0.7), (10.0, 0.3)]

# SVTアルゴリズムでの打ち切り処理
broken = geq(count, c)  # 打ち切り条件
output = branch(broken, NAN, over)  # 打ち切り後はNAN
```

### 依存関係の自動検出

`branch`は共通の依存関係を自動検出し、必要に応じてサンプリングモードに切り替えます：

```python
# ループで同じuを複数回参照
u = Uniform(low=1, high=100).to_dist()
z = Dist.deterministic(0.0)

for idx in range(n):
    condition = geq(threshold[idx], u)
    z = branch(condition, float(idx), z)
    # ✅ branchが依存関係を自動検出
    # ✅ needs_sampling=Trueを設定
    # ✅ @auto_dist()が自動的にサンプリングにフォールバック
```

---

## 3. 集約演算（Aggregation Operations）

| オペレーション | 関数 | クラス | 入力 | 出力 | 機能説明 |
|--------------|------|--------|------|------|---------|
| **最大値インデックス** | `argmax(dists)` | `Argmax` | 分布のリスト | 離散分布 | P(argmax=i) = ∫ f_i(x) ∏_{j≠i} F_j(x) dx |
| **最大値** | `max_op(dists)` | `Max` | 分布のリスト | 分布 | Z = max(X₁, X₂, ..., Xₖ) の分布 |
| **最小値** | `min_op(dists)` | `Min` | 分布のリスト | 分布 | Z = min(X₁, X₂, ..., Xₖ) の分布 |

**数学的定義**:
- **Max**: `F_max(z) = ∏ᵢ F_i(z)`, `f_max(z) = Σᵢ f_i(z) ∏_{j≠i} F_j(z)`
- **Min**: `F_min(z) = 1 - ∏ᵢ (1 - F_i(z))`
- **Argmax**: 各インデックスが最大となる確率の離散分布

**使用例**:
```python
# Report Noisy Max
noisy_values = vector_add(values, noises)
winner_idx = argmax(noisy_values)  # インデックス分布

# 最大値そのもの
max_value = max_op(noisy_values)  # 値の分布
```

---

## 4. ベクトル演算（Vector Operations）

| オペレーション | 関数 | 入力 | 出力 | 機能説明 |
|--------------|------|------|------|---------|
| **ベクトル加算** | `vector_add(xs, ys)` | 2つの分布リスト | 分布リスト | 要素ごとに加算（ブロードキャスト対応） |
| **ベクトルargmax** | `vector_argmax(dists)` | 分布リスト | 離散分布 | `argmax(dists)`のエイリアス |
| **ベクトルmax** | `vector_max(dists)` | 分布リスト | 分布 | `max_op(dists)`のエイリアス |
| **ベクトルmin** | `vector_min(dists)` | 分布リスト | 分布 | `min_op(dists)`のエイリアス |

**ブロードキャスト機能**:
```python
# ベクトル + ベクトル
result = vector_add([d1, d2, d3], [n1, n2, n3])

# ベクトル + スカラー（ブロードキャスト）
result = vector_add([d1, d2, d3], noise)  # すべての要素に同じノイズ
```

---

## 5. 特殊値（Special Values）

| 定数 | 値 | 用途 |
|------|-----|------|
| `NAN` | `float('nan')` | 未定義/打ち切り後の状態を表現（SVTアルゴリズムなど） |
| `PAD` | `-999999.0` | パディング値/未使用スロット |

---

## オペレーション使用パターン

### パターン1: 基本的なノイズ付加
```python
from dpest.operations import add
from dpest.noise import Laplace

# 値にラプラスノイズを追加
Q = Dist.deterministic(10.0)
noisy_Q = add(Q, Laplace(b=1.0).to_dist())
```

### パターン2: 閾値判定（SVTパターン）
```python
from dpest.operations import geq, branch, add, NAN

# 閾値と比較
over = geq(noisy_Q, T)  # {0, 1}分布

# 打ち切り処理
output = branch(broken, NAN, over)  # 打ち切り後はNAN出力
```

### パターン3: Report Noisy Max
```python
from dpest.operations import vector_add, argmax

# ノイズ付加（ベクトル演算）
noisy_values = vector_add(values, noise_dists)

# インデックス分布を取得
winner = argmax(noisy_values)
```

### パターン4: 条件付き混合
```python
from dpest.operations import geq, branch

# 条件分布の作成
cond = geq(x, threshold)

# 条件に応じて異なる分布を混合
result = branch(cond, high_branch, low_branch)
```

---

## 実装上の特徴

| 特徴 | 説明 |
|------|------|
| **FFTベース** | `add`は連続分布の畳み込みにFFTを使用（高速） |
| **混合分布対応** | 離散部分（atoms）と連続部分（density）を別々に処理 |
| **依存関係管理** | `dependencies`属性で変数間の依存を追跡 |
| **サンプリング対応** | すべての演算が`_sample_func`を伝播 |
| **誤差管理** | 格子近似の誤差を`error_bounds`で追跡 |
| **計算グラフ** | `Node`オブジェクトで演算の系譜を記録 |

---

## 完全なAPI一覧

### 基本演算
- `add(x_dist, y_dist)` → `Dist` - 分布の加算
- `affine(x_dist, a, b=0.0)` → `Dist` - アフィン変換

### 比較・条件
- `geq(x_dist, y)` → `Dist` - 大小比較（≥）
- `branch(cond_dist, true_val, false_val)` → `Dist` - 条件分岐（依存関係自動検出）

### 集約
- `argmax(distributions)` → `Dist` - 最大値インデックス
- `max_op(distributions)` → `Dist` - 最大値
- `min_op(distributions)` → `Dist` - 最小値

### ベクトル
- `vector_add(x_list, y_list)` → `List[Dist]` - ベクトル加算
- `vector_argmax(distributions)` → `Dist` - ベクトルargmax
- `vector_max(distributions)` → `Dist` - ベクトルmax
- `vector_min(distributions)` → `Dist` - ベクトルmin

### クラスベースAPI
すべてのオペレーションは対応するクラスの`apply()`メソッドを持ちます：
- `Add.apply(x, y)` / `add(x, y)`
- `Affine.apply(x, a, b)` / `affine(x, a, b)`
- `Compare.geq(x, y)` / `geq(x, y)`
- `Branch.apply(c, t, f)` / `branch(c, t, f)`
- `Argmax.apply(dists)` / `argmax(dists)`
- `Max.apply(dists)` / `max_op(dists)`
- `Min.apply(dists)` / `min_op(dists)`

> **注意**: `Condition.apply()` は内部実装として残っていますが、ユーザーは `branch()` を使用してください。

---

## 全オペレーション総合表

以下は、すべてのオペレーションを一覧できる統合表です。

| カテゴリ | オペレーション | 関数 | クラス | 入力 | 出力 | 機能説明 |
|---------|--------------|------|--------|------|------|---------|
| **基本演算** | 加算 | `add(x, y)` | `Add` | 2つの分布 | `Dist` | Z = X + Y の分布（FFT畳み込み） |
| **基本演算** | アフィン変換 | `affine(x, a, b)` | `Affine` | 分布、係数、定数 | `Dist` | Z = aX + b の分布（ヤコビアン） |
| **比較・条件** | 大小比較 | `geq(x, y)` | `Compare` | 分布/定数 × 2 | `Dist` | X ≥ Y の指示値 {0,1} 分布 |
| **比較・条件** | 条件分岐 | `branch(c, t, f)` | `Branch` | 条件、真値、偽値 | `Dist` | 条件で値を選択（依存関係自動検出） |
| **集約演算** | 最大値インデックス | `argmax(dists)` | `Argmax` | 分布リスト | `Dist` | 最大値のインデックス（離散） |
| **集約演算** | 最大値 | `max_op(dists)` | `Max` | 分布リスト | `Dist` | max(X₁, ..., Xₖ) の分布 |
| **集約演算** | 最小値 | `min_op(dists)` | `Min` | 分布リスト | `Dist` | min(X₁, ..., Xₖ) の分布 |
| **ベクトル演算** | ベクトル加算 | `vector_add(xs, ys)` | - | 分布リスト × 2 | `List[Dist]` | 要素ごと加算（ブロードキャスト可） |
| **ベクトル演算** | ベクトルargmax | `vector_argmax(dists)` | - | 分布リスト | `Dist` | `argmax()`のエイリアス |
| **ベクトル演算** | ベクトルmax | `vector_max(dists)` | - | 分布リスト | `Dist` | `max_op()`のエイリアス |
| **ベクトル演算** | ベクトルmin | `vector_min(dists)` | - | 分布リスト | `Dist` | `min_op()`のエイリアス |
| **特殊値** | 未定義値 | `NAN` | - | - | `float` | `float('nan')` 打ち切り後の状態 |
| **特殊値** | パディング値 | `PAD` | - | - | `float` | `-999999.0` 未使用スロット |

### 関数シグネチャ詳細

```python
# 基本演算
add(x_dist: Dist, y_dist: Dist) → Dist
affine(x_dist: Dist, a: float, b: float = 0.0) → Dist

# 比較・条件
geq(x_dist: Dist, y: Union[Dist, float]) → Dist
branch(cond_dist: Dist, true_val: Union[Dist, float], false_val: Union[Dist, float]) → Dist

# 集約
argmax(distributions: List[Dist]) → Dist
max_op(distributions: List[Dist]) → Dist
min_op(distributions: List[Dist]) → Dist

# ベクトル
vector_add(x_list: List[Dist], y_list: Union[List[Dist], Dist]) → List[Dist]
vector_argmax(distributions: List[Dist]) → Dist
vector_max(distributions: List[Dist]) → Dist
vector_min(distributions: List[Dist]) → Dist
```
