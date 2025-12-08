# OPERATION_DETAILS.md

確率分布演算の実装詳細

このドキュメントでは、各確率分布演算の数学的背景と実装アルゴリズムについて詳細に説明します。

## 目次

1. [加法演算 (Add)](#加法演算-add)
2. [アフィン変換 (Affine)](#アフィン変換-affine)
3. [最大値演算 (Max)](#最大値演算-max)
4. [最小値演算 (Min)](#最小値演算-min)
5. [引数最大値演算 (Argmax)](#引数最大値演算-argmax)
6. [比較演算 (Compare/geq)](#比較演算-comparegeq)
7. [条件分岐演算 (Branch)](#条件分岐演算-branch)

---

## 加法演算 (Add)

### 数式

独立な確率変数 X, Y に対して Z = X + Y の分布を計算：

#### 連続 + 連続
```
f_Z(z) = ∫ f_X(x) f_Y(z-x) dx = (f_X * f_Y)(z)
```
畳み込み積分として計算。

#### 離散 + 離散
```
P(Z=z) = Σ_{x+y=z} P(X=x) P(Y=y)
```
全ての組み合わせを列挙。

#### 離散 + 連続
```
f_Z(z) = Σ_i w_i f_Y(z - a_i)
```
各点質量で連続分布をシフト。

### 実装アルゴリズム

#### 1. 離散+離散の処理
```python
for x_val, x_weight in x_dist.atoms:
    for y_val, y_weight in y_dist.atoms:
        result_atoms.append((x_val + y_val, x_weight * y_weight))
```

#### 2. 離散+連続の処理
```python
# 各点質量に対してシフトした連続分布を作成
for x_val, x_weight in x_dist.atoms:
    shifted_x = y_x + x_val
    shifted_f = y_f * x_weight
    # 統一グリッドに補間して加算
    f_interp = interpolate.interp1d(shifted_x, shifted_f, ...)
    unified_f += f_interp(unified_x)
```

#### 3. 連続+連続の処理（FFT畳み込み）
```python
# 格子を統一
dx = min(dx_X, dx_Y)
unified_x = np.linspace(min_x, max_x, n_points)

# FFTで畳み込み
conv_result = np.convolve(x_unified, y_unified, mode='full') * dx
```

### サポートの計算

Minkowski和を使用：
```
support_Z = support_X ⊕ support_Y
         = {x + y | x ∈ support_X, y ∈ support_Y}
```

---

## アフィン変換 (Affine)

### 数式

Z = aX + b の変換を適用：

#### 連続分布
```
f_Z(z) = (1/|a|) f_X((z-b)/a)
```
逆変換 + ヤコビアン。

#### 点質量
```
(value, weight) ↦ (a·value + b, weight)
```
座標変換のみ。質量は保存。

### 実装アルゴリズム

#### 1. 退化分布の処理
```python
if a == 0:
    return Dist.deterministic(b)  # 全質量がbに集中
```

#### 2. 点質量の変換
```python
result_atoms = [(a * x_val + b, x_weight)
                for x_val, x_weight in x_dist.atoms]
```

#### 3. 連続部分の変換
```python
z_grid = a * x_grid + b
z_f = x_f / abs(a)  # ヤコビアン

# a < 0 の場合は昇順にソート
if a < 0:
    sort_idx = np.argsort(z_grid)
    z_grid = z_grid[sort_idx]
    z_f = z_f[sort_idx]
```

---

## 最大値演算 (Max)

### 数式

独立な確率変数 X₁, X₂, ..., Xₖ に対して Z = max(X₁, X₂, ..., Xₖ)：

#### 累積分布関数
```
F_max(z) = ∏_{i=1}^k F_i(z)
```

#### 確率密度関数
```
f_max(z) = Σ_i f_i(z) ∏_{j≠i} F_j(z)
```

### 実装アルゴリズム

#### 1. 離散分布のMax
```python
def generate_combinations(dist_idx, current_values, current_prob):
    if dist_idx == len(distributions):
        max_val = max(current_values)
        all_combinations.append((max_val, current_prob))
        return

    for value, prob in distributions[dist_idx].atoms:
        generate_combinations(dist_idx + 1,
                            current_values + [value],
                            current_prob * prob)
```

#### 2. 連続分布のMax
```python
# f_max(z) = Σ_i f_i(z) ∏_{j≠i} F_j(z) を計算
f_max = np.zeros(n_grid)

for i in range(len(distributions)):
    f_i = get_density_on_grid(distributions[i], x_grid)

    # 他の分布のCDFの積
    cdf_product = np.ones(n_grid)
    for j in range(len(distributions)):
        if i != j:
            cdf_j = get_cdf_on_grid(distributions[j], x_grid)
            cdf_product *= cdf_j

    f_max += f_i * cdf_product
```

---

## 最小値演算 (Min)

### 数式

Z = min(X₁, X₂, ..., Xₖ)：

#### 累積分布関数
```
F_min(z) = 1 - ∏_i (1 - F_i(z))
```

### 実装アルゴリズム

#### 1. 離散分布のMin
Max演算と同様の組み合わせ生成で、minを計算。

#### 2. 連続分布のMin
```python
# F_min(z) = 1 - ∏_i (1 - F_i(z)) を計算
f_min_cdf = np.ones(n_grid)

for dist in distributions:
    cdf_i = get_cdf_on_grid(dist, x_grid)
    f_min_cdf *= (1 - cdf_i)

f_min_cdf = 1 - f_min_cdf

# 密度を数値微分で計算
f_min_density = np.gradient(f_min_cdf, x_grid)
f_min_density = np.maximum(f_min_density, 0)  # 負値をクリップ
```

---

## 引数最大値演算 (Argmax)

### 数式

Z = argmax(X₁, X₂, ..., Xₖ) で、P(Z=i) を計算：

```
P(argmax = i) = ∫ f_i(x) ∏_{j≠i} F_j(x) dx
```

i番目が最大になる確率は、i番目の密度 × 他がそれ以下になる確率。

### 実装アルゴリズム

#### 1. 連続分布のArgmax
```python
x_grid = target_dist.density['x']
f_target = target_dist.density['f']

# 被積分関数を構築
integrand = f_target.copy()

# ∏_{j≠target} F_j(x) を計算
for other_dist in other_dists:
    cdf_values = compute_cdf_on_grid(other_dist, x_grid)
    integrand *= cdf_values

# 数値積分
return np.trapz(integrand, x_grid)
```

#### 2. 離散分布のArgmax
```python
for target_value, target_weight in target_dist.atoms:
    prob_max = target_weight

    # 他の分布でこの値以下になる確率
    for other_dist in other_dists:
        prob_le = 0.0
        for other_value, other_weight in other_dist.atoms:
            if other_value <= target_value:
                prob_le += other_weight
        prob_max *= prob_le

    total_prob += prob_max
```

---

## 比較演算 (Compare/geq)

### 数式

X ≥ Y の指示値（0または1）の分布を計算：

```
P(X ≥ Y) = ∫∫_{x≥y} f_X(x) f_Y(y) dx dy
```

結果は {0, 1} 上の離散分布（ベルヌーイ分布）。

### 実装アルゴリズム

#### 1. 定数との比較
```python
threshold = float(y)
prob = 0.0

# 離散部分
if x_dist.atoms:
    for val, weight in x_dist.atoms:
        if val >= threshold:
            prob += weight

# 連続部分: x >= threshold の領域で密度を積分
if x_dist.density:
    x_grid = x_dist.density['x']
    f_grid = x_dist.density['f']
    mask = x_grid >= threshold
    if np.any(mask):
        prob += np.trapz(f_grid[mask], x_grid[mask])

return Dist(atoms=[(1, prob), (0, 1-prob)])
```

#### 2. 確率変数との比較
```python
# X ≥ Y を X - Y ≥ 0 に変換
diff = Add.apply(x_dist, Affine.apply(y_dist, -1.0, 0.0))
return Compare.geq(diff, 0.0)
```

---

## 条件分岐演算 (Branch)

### 数式

条件分布 C ∈ {0, 1} に基づいて2つの分布を混合：

```
Z = { T  if C = 1
    { F  if C = 0

f_Z(z) = P(C=1) · f_T(z) + P(C=0) · f_F(z)
```

### 実装アルゴリズム

#### 1. 基本的な混合
```python
p_true = sum(weight for val, weight in cond_dist.atoms if val >= 0.5)
p_false = 1.0 - p_true

# 離散部分の混合
result_atoms = []
for val, weight in true_dist.atoms:
    result_atoms.append((val, p_true * weight))
for val, weight in false_dist.atoms:
    result_atoms.append((val, p_false * weight))

# 連続部分の混合（共通格子に補間）
if true_dist.density and false_dist.density:
    # 共通格子を作成
    x_grid = create_unified_grid(true_dist, false_dist)
    f_true_interp = interpolate.interp1d(...)
    f_false_interp = interpolate.interp1d(...)
    f_mix = p_true * f_true_interp(x_grid) + p_false * f_false_interp(x_grid)
```

#### 2. 依存関係の処理
条件分布と分岐先の分布が共通の確率変数に依存する場合、条件付き確率を用いて正しく計算する必要があります（実装では `dependencies` を追跡）。

---

## 例：具体的な計算

### Lap(1) + Lap(1) の計算

```python
# 1. 両方とも格子 [-7, 7] で近似
x1 = np.linspace(-7, 7, 1000)
x2 = np.linspace(-7, 7, 1000)
f1 = 0.5 * np.exp(-np.abs(x1))
f2 = 0.5 * np.exp(-np.abs(x2))

# 2. FFT畳み込み
result_f = np.convolve(f1, f2, mode='full') * dx
result_x = np.linspace(-14, 14, len(result_f))

# 3. 正規化
result_f = result_f / (np.sum(result_f) * dx)
```

この詳細な実装により、差分プライバシー機構の正確な確率分布計算が可能になります。
