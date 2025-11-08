# dpest アルゴリズム実装一覧

このドキュメントは、`dpest/algorithms/`ディレクトリに実装されているすべての差分プライバシーアルゴリズムの一覧です。

## アルゴリズム分類

### 1. Sparse Vector Technique (SVT) 系

| アルゴリズム | 関数 | 特徴 | 使用オペレーション |
|------------|------|------|-----------------|
| **SVT1** | `svt1()` | 基本SVT、閾値ノイズ1回 | `add`, `affine`, `mux`, `geq` |
| **SVT2** | `svt2()` | TRUE毎に閾値再サンプリング | `add`, `affine`, `mux`, `geq` |
| **SVT3** | `svt3()` | TRUEでノイズ付き値出力 | `add`, `affine`, `mux`, `geq` |
| **SVT4** | `svt4()` | eps分割が異なる（eps1=eps/4） | `add`, `affine`, `mux`, `geq` |
| **SVT5** | `svt5()` | クエリにノイズなし（非DP） | `affine`, `geq` |
| **SVT6** | `svt6()` | カウンタ・打ち切りなし | `add`, `affine`, `geq` |
| **NumericalSVT** | `numerical_svt()` | 比較用・出力用で別ノイズ | `add`, `affine`, `mux`, `geq` |

**共通パターン**:
```python
# 閾値にノイズ
T = affine(Laplace(b=...).to_dist(), 1.0, t)

# クエリにノイズ
noisy_Q = add(Q, Laplace(b=...).to_dist())

# 閾値判定
over = geq(noisy_Q, T)

# 打ち切り処理
output = mux(broken, NAN, over)
```

---

### 2. Report Noisy Max 系

| アルゴリズム | 関数 | ノイズ種類 | 出力 | 使用オペレーション |
|------------|------|----------|------|-----------------|
| **Report Noisy Max 1** | `report_noisy_max1()` | Laplace | インデックス | `vector_add`, `vector_argmax` |
| **Report Noisy Max 2** | `report_noisy_max2()` | Exponential | インデックス | `vector_add`, `vector_argmax` |
| **Report Noisy Max 3** | `report_noisy_max3()` | Laplace | 最大値 | `vector_add`, `vector_max` |
| **Report Noisy Max 4** | `report_noisy_max4()` | Exponential | 最大値 | `vector_add`, `vector_max` |

**共通パターン**:
```python
# ノイズ付加
noise_dists = create_laplace_noise(b=2/eps, size=len(values))
noisy_values = vector_add(values, noise_dists)

# 最大値のインデックス or 値
result = vector_argmax(noisy_values)  # または vector_max()
```

---

### 3. Noisy Histogram 系

| アルゴリズム | 関数 | スケール | 使用オペレーション |
|------------|------|---------|-----------------|
| **Noisy Histogram 1** | `noisy_hist1()` | `b = 1/eps` | `vector_add` |
| **Noisy Histogram 2** | `noisy_hist2()` | `b = eps` | `vector_add` |

**パターン**:
```python
# 各バケットに独立ラプラスノイズ
noise_dists = create_laplace_noise(b=1/eps, size=len(values))
return vector_add(values, noise_dists)
```

---

### 4. Laplace Mechanism 系

| アルゴリズム | 関数 | 特徴 | 使用オペレーション |
|------------|------|------|-----------------|
| **Laplace Vector** | `laplace_vec()` | ベクトル各要素にノイズ | `vector_add` |
| **Laplace Parallel** | `laplace_parallel()` | 同じ値に独立ノイズを並列適用 | `add` |

**Laplace Parallel パターン**:
```python
# スカラー値を取得
scalar = expect_single_value(values, "laplace_parallel")
base = Dist.deterministic(scalar)

# n_parallel個の独立ノイズ
noise_list = create_laplace_noise(b=1/eps_each, size=n_parallel)
return [add(base, n) for n in noise_list]
```

---

### 5. RAPPOR 系

| アルゴリズム | 関数 | 特徴 | 使用オペレーション |
|------------|------|------|-----------------|
| **One-time RAPPOR** | `one_time_rappor()` | 1回限りの確率的応答 | `condition` |
| **RAPPOR** | `rappor()` | 永続的確率的応答 | `condition` |

**パターン**:
```python
# 確率的ランダム化（conditionを使用）
cond_randomize = Dist.from_atoms([(1.0, f), (0.0, 1.0 - f)])
random_bit = condition(cond_flip, bit_one, bit_zero)
perm = condition(cond_randomize, random_bit, base)
```

---

### 6. その他

| アルゴリズム | 関数 | 特徴 | 使用オペレーション |
|------------|------|------|-----------------|
| **Noisy Max Sum** | `noisy_max_sum()` | 2ベクトルの noisy max の和 | `vector_add`, `vector_max`, `add` |

**パターン**:
```python
def noisy_max(vec: List[Dist]) -> Dist:
    noise = create_laplace_noise(b=1/eps, size=len(vec))
    noisy_vec = vector_add(vec, noise)
    return vector_max(noisy_vec)

max1 = noisy_max(vec1)
max2 = noisy_max(vec2)
return add(max1, max2)
```

---

## オペレーション使用統計

| オペレーション | 使用回数 | 主な用途 |
|--------------|---------|---------|
| `add` | 9 | ノイズ付加、値の加算 |
| `affine` | 7 | 閾値のシフト |
| `mux` | 5 | SVT系の打ち切り処理 |
| `geq` | 7 | 閾値判定 |
| `vector_add` | 9 | ベクトルへのノイズ付加 |
| `vector_argmax` | 2 | Report Noisy Max（インデックス） |
| `vector_max` | 3 | Report Noisy Max（値） |
| `condition` | 2 | RAPPOR系のランダム化 |

---

## アルゴリズム実装の共通パターン

### パターン1: SVT系 - 打ち切り処理

```python
from dpest.operations import add, affine, mux, geq
from dpest.operations import NAN

# 初期化
count = Dist.deterministic(0.0)
broken = Dist.deterministic(0.0)

for Q in queries:
    # ノイズ付加と判定
    noisy_Q = add(Q, Laplace(b=...).to_dist())
    over = geq(noisy_Q, T)

    # 打ち切り後はNAN
    out_i = mux(broken, NAN, over)

    # カウンタ更新
    inc = mux(broken, 0, over)
    count = add(count, inc)
    broken = geq(count, c)
```

### パターン2: Report Noisy Max系 - ベクトル演算

```python
from dpest.operations import vector_add, vector_argmax

# ノイズ付加
noise_dists = create_laplace_noise(b=2/eps, size=len(values))
noisy_values = vector_add(values, noise_dists)

# argmax
return vector_argmax(noisy_values)
```

### パターン3: RAPPOR系 - 確率的ランダム化

```python
from dpest.operations import condition

# 確率分布の作成
cond = Dist.from_atoms([(1.0, p), (0.0, 1.0 - p)])

# 条件付き混合
result = condition(cond, dist_if_true, dist_if_false)
```

---

## アルゴリズム選択ガイド

| 目的 | 推奨アルゴリズム | 理由 |
|------|---------------|------|
| 閾値判定（複数クエリ） | SVT1, SVT2 | 基本的なSVT |
| 閾値判定 + 値出力 | SVT3, NumericalSVT | ノイズ付き値を返す |
| 最大値のインデックス | Report Noisy Max 1/2 | argmaxを返す |
| 最大値そのもの | Report Noisy Max 3/4 | 最大値を返す |
| ヒストグラム公開 | Noisy Histogram 1 | 標準的な設定 |
| ベクトルへのノイズ付加 | Laplace Vec | 各要素独立にノイズ |
| 同一値の複数公開 | Laplace Parallel | 並列合成 |
| ローカルDP（ビット応答） | RAPPOR | Bloom filter + randomization |

---

## 全アルゴリズム一覧（シグネチャ付き）

```python
# SVT系
svt1(queries: List[Dist], eps: float = 0.1, t: float = 0.5, c: int = 1) -> List[Dist]
svt2(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]
svt3(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]
svt4(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]
svt5(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]
svt6(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]
numerical_svt(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]

# Report Noisy Max系
report_noisy_max1(values: List[Dist], eps: float) -> Dist
report_noisy_max2(values: List[Dist], eps: float) -> Dist
report_noisy_max3(values: List[Dist], eps: float) -> Dist
report_noisy_max4(values: List[Dist], eps: float) -> Dist

# Noisy Histogram系
noisy_hist1(values: List[Dist], eps: float) -> List[Dist]
noisy_hist2(values: List[Dist], eps: float) -> List[Dist]

# Laplace Mechanism系
laplace_vec(values: List[Dist], eps: float) -> List[Dist]
laplace_parallel(values: List[Dist], eps_each: float, n_parallel: int) -> List[Dist]

# RAPPOR系
one_time_rappor(values: List[Dist], eps: float, n_hashes: int = 4,
                filter_size: int = 20, f: float = 0.95) -> List[Dist]
rappor(values: List[Dist], eps: float, n_hashes: int = 4, filter_size: int = 20,
       f: float = 0.75, p: float = 0.45, q: float = 0.55) -> List[Dist]

# その他
noisy_max_sum(values: List[Dist], eps: float = 0.1, split_index: int | None = None) -> Dist
```
