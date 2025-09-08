# OPERATION.md

確率分布演算の実装詳細

このドキュメントでは、差分プライバシーライブラリで実装された各確率分布演算の数学的背景と計算手法について詳細に説明します。

## 目次

1. [基本概念](#基本概念)
2. [加法演算 (Add)](#加法演算-add)
3. [アフィン変換 (Affine)](#アフィン変換-affine)
4. [最大値演算 (Max)](#最大値演算-max)
5. [最小値演算 (Min)](#最小値演算-min)
6. [引数最大値演算 (Argmax)](#引数最大値演算-argmax)
7. [累積和演算 (PrefixSum)](#累積和演算-prefixsum)
8. [依存する確率変数への対応](#依存する確率変数への対応)
9. [実装上の注意点](#実装上の注意点)

---

## 基本概念

### 確率分布の表現

本ライブラリでは、確率分布を以下の混合形式で表現します：

```
Dist = (atoms, density, support, error_bounds)
```

- **atoms**: 点質量（離散分布）`[(value, weight), ...]`
- **density**: 連続密度の格子近似 `{'x': grid_x, 'f': grid_f, 'dx': dx}`
- **support**: サポート区間のリスト `[Interval(low, high), ...]`
- **error_bounds**: 数値誤差の上界

### 数値計算の方針

- **FFT**: 連続分布同士の畳み込み
- **補間**: 格子の統一と再サンプリング
- **Minkowski和**: サポート区間の計算
- **数値積分**: 台形法による近似

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

区間の場合：
```
[a, b] ⊕ [c, d] = [a+c, b+d]
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

#### 4. サポートの変換
```python
if a > 0:
    new_interval = Interval(a * interval.low + b, a * interval.high + b)
else:
    new_interval = Interval(a * interval.high + b, a * interval.low + b)
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

#### 3. CDFの数値計算
```python
for i, x in enumerate(x_grid):
    mask = dist_x <= x
    if np.any(mask):
        cdf_values[i] = np.trapz(dist_f[mask], dist_x[mask])

# 点質量の寄与を追加
if dist.atoms:
    for atom_value, atom_weight in dist.atoms:
        mask = x_grid >= atom_value
        cdf_values[mask] += atom_weight
```

---

## 最小値演算 (Min)

### 数式

Z = min(X₁, X₂, ..., Xₖ)：

#### 累積分布関数
```
1 - F_min(z) = ∏_i (1 - F_i(z))
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

#### 3. CDFの格子上計算
台形法による累積積分：
```python
for i, x in enumerate(x_grid):
    mask = dist_x <= x
    if np.any(mask):
        cdf_values[i] = np.trapz(dist_f[mask], dist_x[mask])
```

---

## 累積和演算 (PrefixSum)

複数の独立な確率変数 $X_1, X_2, \ldots, X_n$ に対し、各ステップの累積和
$S_k = \sum_{i=1}^k X_i$ の分布を `Add` 演算を用いて逐次計算します。
結果は各ステップの分布を要素とするリストとして返されます。

---

## 依存する確率変数への対応

本ライブラリの演算は基本的に入力の独立性を仮定しますが、
`Add` や `Max`、`Min`、`Argmax` では `joint_samples` 引数を受け取り、
サンプルに基づく近似計算を行うことができます。
依存関係がある場合は共通の乱数サンプルを提供してください。

```python
samples = np.random.multivariate_normal(mean, cov, size=1000)
res = Add.apply(x_dist, y_dist, joint_samples=samples)
```

---

## 実装上の注意点

### 1. 数値安定性

- **微小値の処理**: 1e-10 未満は0とみなす
- **正規化**: 計算後に確率質量を1に正規化
- **クリッピング**: CDF値を[0,1]にクリップ

### 2. 格子の統一

```python
# 統一格子の作成
min_support = min(all_supports)
max_support = max(all_supports)
unified_x = np.linspace(min_support, max_support, n_grid)

# 補間による統一
f_interp = interpolate.interp1d(dist_x, dist_f, 
                               bounds_error=False, fill_value=0.0)
```

### 3. メモリ効率

- **スパース表現**: 非ゼロ要素のみ保存
- **適応的格子**: 必要な解像度のみ使用
- **点質量の統合**: 同じ値をマージ

### 4. 誤差評価

各演算で以下の誤差が蓄積：
- **err_trunc**: 格子の切り捨て誤差
- **err_interp**: 補間誤差
- **err_quad**: 数値積分誤差

### 5. 境界条件

- **サポート外**: 確率密度は0
- **無限大**: 適切な範囲で切り捨て
- **特異点**: 適切な数値的処理

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

結果: Lap(1) + Lap(1) ≈ Lap(1) の畳み込み

この詳細な実装により、差分プライバシー機構の正確な確率分布計算が可能になります。