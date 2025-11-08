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
| **条件分岐** | `condition(c, t, f)` | `Condition` | 条件分布、真分布、偽分布 | 混合分布 | P(c=1)·t + P(c=0)·f の混合分布 |
| **マルチプレクサ** | `mux(c, t, f)` | `MUX` | 条件分布、真値、偽値 | 分布 | 条件に基づいて値を選択（定数も可） |

### `condition` vs `mux` の違い

> **なぜ2つの関数が存在するのか？**
>
> `condition`と`mux`は技術的には同じ混合分布を生成しますが、**使用目的と意図が異なります**：
> - **`condition`**: 確率論的な概念（混合分布）を明示的に表現
> - **`mux`**: プログラミング的な概念（値の選択）を簡潔に表現
>
> この区別により、アルゴリズムの意図が**コードから読み取りやすく**なります。
> 実際、SVT系では`mux`、RAPPOR系では`Condition`が使われています。

#### `condition(cond_dist, true_dist, false_dist)` - 条件付き混合分布

**目的**: 2つの確率分布を条件に基づいて**混合**する

**動作原理**:
- 条件分布`cond_dist`が真（値≥0.5）となる確率を`p`とする
- 結果は`p × true_dist + (1-p) × false_dist`の混合分布
- 両方の分布の可能性が重ね合わされた**確率的な混合**

**数学的表現**:
```
P(Z ∈ A) = P(条件=真) × P(true_dist ∈ A) + P(条件=偽) × P(false_dist ∈ A)
```

**使用場面**:
- 条件が確率的に決まる場合
- 両方の枝の可能性を保持したい場合
- 確率分布として扱いたい場合

**例**:
```python
# 閾値を超える確率が70%の場合
cond = geq(noisy_value, threshold)  # P(cond=1) = 0.7, P(cond=0) = 0.3

high = Dist.deterministic(100.0)
low = Dist.deterministic(10.0)

result = condition(cond, high, low)
# 結果: 70%の確率で100付近、30%の確率で10付近の混合分布
# atoms = [(100.0, 0.7), (10.0, 0.3)]
```

---

#### `mux(cond_dist, true_val, false_val)` - マルチプレクサ

**目的**: 条件に基づいて2つの値から1つを**選択**する（定数も可能）

**動作原理**:
- 内部的には`condition()`を使用
- ただし、定数値を自動的に`Dist.deterministic()`に変換
- センチネル値（`NAN`, `PAD`など）の処理に便利

**違い**:
- `condition()`と同じ混合分布を返すが、使いやすさを重視
- 定数を直接渡せる（`NAN`などの特殊値に便利）
- 計算グラフ上のノードタイプが`'MUX'`（デバッグしやすい）

**使用場面**:
- センチネル値（`NAN`, `PAD`）を扱う場合
- 定数値との混合が必要な場合
- 意味的に「選択」を表現したい場合（可読性）

**例**:
```python
# SVTアルゴリズムでの打ち切り処理
broken = geq(count, c)  # 打ち切り条件

# 打ち切り後はNAN、そうでなければ比較結果
output = mux(broken, NAN, comparison_result)
# brokenがTrue（確率1.0）なら → NAN（確率1.0）
# brokenがFalse（確率1.0）なら → comparison_result
# brokenが確率的（例: 0.3）なら → 30%でNAN、70%でcomparison_result
```

---

### 具体的な違いの例

```python
from dpest.core import Dist
from dpest.operations import condition, mux, geq

# 準備
x = Dist.deterministic(15.0)
threshold = Dist.deterministic(10.0)
cond = geq(x, threshold)  # P(cond=1) = 1.0（確定的にTrue）

high_dist = Dist.deterministic(100.0)
low_dist = Dist.deterministic(50.0)

# 1. condition - 両方とも分布を渡す必要がある
result1 = condition(cond, high_dist, low_dist)
# atoms = [(100.0, 1.0)]（確率1.0で100）

# 2. mux - 定数を直接渡せる
result2 = mux(cond, 100.0, 50.0)  # ← 定数でOK
# atoms = [(100.0, 1.0)]（同じ結果）

# 3. センチネル値の場合
result3 = mux(cond, NAN, 0.0)  # ← NANを直接渡せる
# atoms = [(nan, 1.0)]（確率1.0でNAN）

# 4. conditionでNANを使う場合（冗長）
result4 = condition(cond, Dist.deterministic(NAN), Dist.deterministic(0.0))
# 同じ結果だが、muxの方が簡潔
```

---

### 選択ガイドライン

| 状況 | 推奨 | 理由 |
|------|------|------|
| 両方とも既に`Dist`オブジェクト | `condition` | 意図が明確（確率的混合） |
| 定数値（特に`NAN`, `PAD`）を使う | `mux` | 簡潔で読みやすい |
| SVT系アルゴリズム | `mux` | センチネル値の処理に最適 |
| RAPPOR系アルゴリズム | `condition` | 確率的ランダム化を表現 |
| 確率的な分岐ロジック | `condition` | 確率論的な意味が明確 |
| 値の選択（if-else的） | `mux` | プログラミング的で直感的 |

### 実際の使用例（dpest内のアルゴリズム）

**SVT1アルゴリズムでの`mux`の使用**:
```python
# dpest/algorithms/svt1.py
broken = geq(count, c)  # 打ち切り条件

# 打ち切り後はNANを出力
out_i = mux(broken, NAN, over)  # センチネル値を直接使用

# カウンタ更新（打ち切り後は加算しない）
inc = mux(broken, 0, over)  # 定数0を直接使用
```

**One-time RAPPORでの`Condition`の使用**:
```python
# dpest/algorithms/one_time_rappor.py
cond_randomize = Dist.from_atoms([(1.0, f), (0.0, 1.0 - f)])
cond_flip = Dist.from_atoms([(1.0, 0.5), (0.0, 0.5)])

# 確率的なランダム化（明示的な確率分布の混合）
random_bit = Condition.apply(cond_flip, bit_one, bit_zero)
perm = Condition.apply(cond_randomize, random_bit, base)
```

### 統合しない理由

技術的には統合可能ですが、**意味的な区別を保持**することで以下のメリットがあります：

1. **コードの可読性**: アルゴリズムの意図が一目で分かる
   - `mux` → 「条件による値の選択」
   - `condition` → 「確率的な混合」

2. **実装パターンの明確化**:
   - SVT系 → 打ち切り処理に`mux`を使うパターン
   - RAPPOR系 → ランダム化に`Condition`を使うパターン

3. **保守性**: 将来的に異なる最適化が可能
   - `mux`は定数の場合に最適化可能
   - `condition`は混合分布の数学的性質を活用可能

**使用例**:
```python
# 閾値との比較
over_threshold = geq(noisy_Q, T)  # {0: P(Q<T), 1: P(Q≥T)}

# 条件付き出力（分布の混合）
result = condition(over_threshold, high_value, low_value)

# センチネル値の処理（定数を直接使用）
output = mux(broken, NAN, over_threshold)  # 打ち切り後はNAN
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
from dpest.operations import geq, mux, add

# 閾値と比較
over = geq(noisy_Q, T)  # {0, 1}分布

# 打ち切り処理
output = mux(broken, NAN, over)  # 打ち切り後はNAN出力
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
from dpest.operations import geq, condition

# 条件分布の作成
cond = geq(x, threshold)

# 条件に応じて異なる分布を混合
result = condition(cond, high_branch, low_branch)
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
- `condition(cond_dist, true_dist, false_dist)` → `Dist` - 条件付き混合
- `mux(cond_dist, true_val, false_val)` → `Dist` - マルチプレクサ

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
- `Condition.apply(c, t, f)` / `condition(c, t, f)`
- `MUX.apply(c, t, f)` / `mux(c, t, f)`
- `Argmax.apply(dists)` / `argmax(dists)`
- `Max.apply(dists)` / `max_op(dists)`
- `Min.apply(dists)` / `min_op(dists)`

---

## 全オペレーション総合表

以下は、すべてのオペレーションを一覧できる統合表です。

| カテゴリ | オペレーション | 関数 | クラス | 入力 | 出力 | 機能説明 |
|---------|--------------|------|--------|------|------|---------|
| **基本演算** | 加算 | `add(x, y)` | `Add` | 2つの分布 | `Dist` | Z = X + Y の分布（FFT畳み込み） |
| **基本演算** | アフィン変換 | `affine(x, a, b)` | `Affine` | 分布、係数、定数 | `Dist` | Z = aX + b の分布（ヤコビアン） |
| **比較・条件** | 大小比較 | `geq(x, y)` | `Compare` | 分布/定数 × 2 | `Dist` | X ≥ Y の指示値 {0,1} 分布 |
| **比較・条件** | 条件分岐 | `condition(c, t, f)` | `Condition` | 条件、真、偽 | `Dist` | P(c=1)·t + P(c=0)·f 混合分布 |
| **比較・条件** | マルチプレクサ | `mux(c, t, f)` | `MUX` | 条件、真値、偽値 | `Dist` | 条件で値を選択（定数可） |
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
condition(cond_dist: Dist, true_dist: Dist, false_dist: Dist) → Dist
mux(cond_dist: Dist, true_val: Union[Dist, float], false_val: Union[Dist, float]) → Dist

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
