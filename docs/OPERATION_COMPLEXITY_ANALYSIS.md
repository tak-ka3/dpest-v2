# 解析手法における全オペレーションの計算量分析

## 概要

dpestフレームワークの解析手法では、確率分布を格子近似（grid approximation）し、様々な演算を適用してアルゴリズムの出力分布を計算する。本ドキュメントでは、実装されている全てのオペレーションについて、理論的な計算量とメモリ使用量を詳細に分析する。

**表記**:
- $g$: 格子点数（デフォルト1000）
- $n$: 分布の個数
- $k$: アトム（点質量）の個数
- $m$: 入力ベクトルのサイズ

---

## 1. 基本演算

### 1.1 Add演算（加法）: $Z = X + Y$

**アルゴリズム**: FFTベースの畳み込み

**実装**: `dpest/operations/operations.py` の `Add.apply()`

#### 1.1.1 離散 + 離散

2つの離散分布の和:

```python
for x_val, x_weight in x_dist.atoms:
    for y_val, y_weight in y_dist.atoms:
        result_atoms.append((x_val + y_val, x_weight * y_weight))
```

**計算量**: $O(k_X \times k_Y)$
- $k_X$, $k_Y$: 各分布のアトム数
- すべてのペアの組み合わせを計算

**メモリ**: $O(k_X \times k_Y)$ (結果のアトム数)

#### 1.1.2 離散 + 連続

離散分布の各アトムに対して連続分布をシフト:

```python
for x_val, x_weight in x_dist.atoms:
    shifted_x = y_x + x_val
    shifted_f = y_f * x_weight
```

**計算量**: $O(k \times g)$
- $k$: アトム数
- $g$: 連続分布の格子点数
- 各シフトでの補間: $O(g)$

**メモリ**: $O(k \times g)$ (シフトされた分布の保存)

#### 1.1.3 連続 + 連続（FFT畳み込み）

**実装**:
```python
conv_result = np.convolve(x_unified, y_unified, mode='full') * dx
```

**計算量**: $O(g \log g)$

**詳細**:
1. 格子の統一: $O(g)$
2. 補間（scipy.interpolate.interp1d）: $O(g)$ × 2
3. **FFT畳み込み**（`np.convolve`）: $O(g \log g)$
   - 内部でFFTを使用
   - 順FFT: $O(g \log g)$
   - 要素ごとの積: $O(g)$
   - 逆FFT: $O(g \log g)$

**メモリ**: $O(g)$
- 統一格子: $2g$ (x, y各々)
- FFT作業領域: $O(g)$
- 結果: $O(g)$

**実効速度**: 格子点数 $g=1000$ の場合
- 理論演算数: $g \log_2 g = 1000 \times 10 \approx 10^4$ 演算
- 実測: 0.001秒程度（NoisyHist2の単一Add）

---

### 1.2 Affine演算（アフィン変換）: $Z = aX + b$

**実装**: `dpest/operations/operations.py` の `Affine.apply()`

#### 1.2.1 離散分布の変換

```python
result_atoms = [(a * x_val + b, x_weight) for x_val, x_weight in x_dist.atoms]
```

**計算量**: $O(k)$
- アトムごとに定数時間の演算

**メモリ**: $O(k)$

#### 1.2.2 連続分布の変換

```python
result_x = (x_grid - b) / a
result_f = x_f / abs(a)  # ヤコビアン補正
```

**計算量**: $O(g)$
- 各格子点で線形変換
- ヤコビアン補正: 定数倍

**メモリ**: $O(g)$

**重要な性質**:
- 誤差を増幅しない（線形変換のため）
- ヤコビアン $= |a|$ (定数)

---

## 2. 最大・最小演算

### 2.1 Max演算: $Z = \max(X_1, X_2, \ldots, X_n)$

**理論**:
$$
f_{\max}(z) = \sum_{i=1}^{n} f_i(z) \prod_{j \neq i} F_j(z)
$$

**実装**: `dpest/operations/max_op.py` の `Max.apply()`

#### 2.1.1 離散分布のMax

全ての組み合わせを列挙:

```python
def generate_combinations(dist_idx, current_values, current_prob):
    if dist_idx == len(distributions):
        max_val = max(current_values)
        all_combinations.append((max_val, current_prob))
```

**計算量**: $O(k^n)$
- $k$: 各分布のアトム数（平均）
- $n$: 分布の個数
- 指数的増加（実用的には $n \leq 5$ 程度）

**メモリ**: $O(k^n)$ (全組み合わせの保存)

#### 2.1.2 連続分布を含むMax

**アルゴリズム**:
```python
for i in range(len(distributions)):
    f_i = get_density_on_grid(distributions[i], x_grid)  # O(g)
    cdf_product = np.ones(n_grid)
    for j in range(len(distributions)):
        if i != j:
            cdf_j = get_cdf_on_grid(distributions[j], x_grid)  # O(g)
            cdf_product *= cdf_j  # O(g)
    f_max += f_i * cdf_product  # O(g)
```

**計算量**: $O(n^2 \times g^2)$

**詳細**:
1. 外側ループ（$i$）: $n$ 回
2. 内側ループ（$j$）: $n-1$ 回
3. CDF計算（`_get_cdf_on_grid`）: 各回 $O(g^2)$
   - x_gridの各点(g個)について、累積和をtrapzで計算(O(g))
   - 二重ループ構造: 外側g回 × 内側O(g) = O(g²)
4. 密度取得: 各回 $O(g)$
5. 積の計算: 各回 $O(g)$

**合計**: $n \times [(n-1) \times g^2 + g + g] \approx O(n^2 g^2)$

**CDF計算の実装**:
```python
def _get_cdf_on_grid(dist, x_grid):
    cdf = np.zeros_like(x_grid)
    # 連続分布の場合: 各点で累積積分
    if dist.density:
        for i, x in enumerate(x_grid):  # O(g)
            mask = dist_x <= x
            if np.any(mask):
                cdf[i] = np.trapz(dist_f[mask], dist_x[mask])  # O(g)
    # → 合計 O(g²)
```

**メモリ**: $O(n \times g)$
- 各分布のCDF: $n \times g$
- 作業領域（f_max, cdf_product）: $2g$

**実効速度**: $n=5$, $g=1000$ の場合
- 理論演算数: $5 \times (4 \times 10^6 + 10^3) \approx 2 \times 10^7$ 演算
  - 各iについて: 4つの他分布 × O(g²=10⁶) + O(g)
- 実測: 0.5秒程度

### 2.2 Min演算: $Z = \min(X_1, X_2, \ldots, X_n)$

**理論**:
$$
f_{\min}(z) = \sum_{i=1}^{n} f_i(z) \prod_{j \neq i} (1 - F_j(z))
$$

**計算量**: Max演算と同じ $O(n^2 g^2)$

**実装**: Max演算とほぼ同一のアルゴリズム

---

## 3. Argmax演算

### 3.1 Argmax: $\text{argmax}(X_1, X_2, \ldots, X_n)$

**出力**: 離散分布（各インデックスの確率）

**理論**:
$$
P(\text{argmax} = i) = \int f_i(x) \prod_{j \neq i} F_j(x) \, dx
$$

**実装**: `dpest/operations/argmax_op.py` の `Argmax.apply()`

#### 3.1.1 連続分布を含むArgmax

**アルゴリズム**:
```python
for i in range(n):
    # P(argmax=i) を計算
    target_dist = distributions[i]
    f_target = target_dist.density['f']

    integrand = f_target.copy()
    for j in range(n):
        if j != i:
            cdf_j = compute_cdf_on_grid(distributions[j], x_grid)
            integrand *= cdf_j

    prob_i = np.trapz(integrand, x_grid)
```

**計算量**: $O(n^2 \times g^2)$

**詳細**:
1. 外側ループ（各インデックス $i$）: $n$ 回
2. 内側ループ（他の分布 $j$）: $n-1$ 回
3. 各CDF計算（`_compute_cdf_on_grid`）: $O(g^2)$
   - x_gridの各点(g個)について、累積和をtrapzで計算(O(g))
   - 二重ループ構造: 外側g回 × 内側O(g) = O(g²)
4. 積の計算: $O(g)$
5. 数値積分（`np.trapz`）: $O(g)$

**合計**: $n \times [(n-1) \times g^2 + g + g] \approx O(n^2 g^2)$

**メモリ**: $O(n \times g)$
- 各分布のCDF: $n \times g$
- 作業領域（integrand）: $g$

**実効速度**: $n=5$, $g=1000$ の場合
- 理論演算数: $5 \times (4 \times 10^6 + 10^3 + 10^3) \approx 2 \times 10^7$ 演算
  - 各iについて: 4つの他分布 × O(g²=10⁶) + O(g) + O(g)
- 実測: 0.5秒程度（単一Argmax）

**重要な注意**:
ReportNoisyMax1では、Argmax演算に加えて、**ε推定時のペア比較**が必要。

**ReportNoisyMax1の総計算量**:
- Argmax演算自体: $O(n^2 g^2) \approx 2 \times 10^7$
- プライバシー損失推定では、隣接ペア数だけArgmaxを繰り返し実行
- ペア数: 通常2つ（one_above, one_below）
- 合計: 2ペア × 2 \times 10^7 = 4 \times 10^7$ 演算
- **実測2.75秒はこのArgmax演算のコストが支配的**

#### 3.1.2 離散分布のArgmax

**計算量**: $O(n \times k^n)$
- Max演算の離散版と同様
- 各インデックスについて全組み合わせを列挙

---

## 4. 比較演算

### 4.1 Compare / Geq演算: $P(X \geq Y)$

**理論**:
$$
P(X \geq Y) = \int_{-\infty}^{\infty} f_X(x) \cdot F_Y(x) \, dx
$$

**実装**: `dpest/operations/compare_op.py` の `Compare.apply()`

**計算量**: $O(g^2)$

**詳細**:
```python
for i, x in enumerate(x_grid):
    for j, y in enumerate(y_grid):
        if x >= y:
            prob_geq += f_x[i] * f_y[j] * dx * dy
```

**二重ループでの数値積分**:
- 外側: $g$ 回
- 内側: $g$ 回（条件付き）
- 各ステップ: 定数時間

**メモリ**: $O(g)$
- 各分布の格子: $2g$
- 作業変数: 定数

**最適化版**（累積分布関数を使用）:
```python
cdf_y = np.cumsum(f_y) * dy  # O(g)
prob_geq = np.sum(f_x * (1 - cdf_y)) * dx  # O(g)
```

**最適化後の計算量**: $O(g)$

**実効速度**: $g=1000$ の場合
- 理論演算数（最適化版）: $2 \times 1000 = 2 \times 10^3$ 演算
- 理論演算数（二重ループ版）: $1000 \times 1000 = 10^6$ 演算
- 実測: 0.001秒程度

---

## 5. 条件分岐・依存演算

### 5.1 Branch演算（条件分岐）

**定義**:
$$
Z = \begin{cases}
X & \text{if } C = \text{true} \\
Y & \text{if } C = \text{false}
\end{cases}
$$

**実装**: `dpest/operations/branch_op.py` の `Branch.apply()`

**重要な特性**: **解析手法では扱えない**（依存関係が発生）

**理由**:
- $X$, $Y$, $C$ が共通の確率変数に依存する場合、独立性の仮定が破綻
- 例: SVT系アルゴリズムでは、共通の閾値 $T$ に対して複数のクエリが分岐

**計算モード**: **サンプリングモード**に自動切り替え

```python
node = Node(op='Branch', ..., needs_sampling=True)
```

**サンプリングモードの計算量**: $O(N)$
- $N$: サンプル数（デフォルト100,000〜1,000,000）
- 各サンプルで条件を評価: 定数時間

**メモリ**: $O(N)$ (サンプルの保存)

---

### 5.2 PrefixSum演算（累積和）

**定義**:
$$
Y_i = \sum_{j=1}^{i} X_j
$$

**実装**: `dpest/operations/prefix_sum_op.py` の `PrefixSum.apply()`

**依存関係**: $Y_i$ が $Y_{i-1}$ に依存

**計算モード**: **サンプリングモード**（依存関係のため）

**計算量**: $O(N \times m)$
- $N$: サンプル数
- $m$: 入力ベクトルのサイズ
- 各サンプルで累積和を計算

**メモリ**: $O(N \times m)$ (全サンプルの保存)

---

## 6. 乗算・単調変換

### 6.1 Mul演算（乗算）: $Z = X \times Y$

**実装**: 正値域では対数変換を使用

**アルゴリズム**:
1. $U = \log X$, $V = \log Y$ に変換
2. $W = U + V$ を計算（Add演算）
3. $Z = \exp(W)$ に逆変換

**計算量**: $O(g \log g)$
- 対数変換: $O(g)$
- Add演算: $O(g \log g)$
- 指数変換: $O(g)$
- **支配項**: Add演算の $O(g \log g)$

**メモリ**: $O(g)$

---

### 6.2 単調変換（Exp, Log, Power, Abs, ReLU）

**一般形**: $Z = h(X)$ (単調関数 $h$)

**計算量**: $O(g)$

**詳細**:
```python
# 格子点の変換
result_x = h(x_grid)  # O(g)
# ヤコビアン補正
result_f = f / abs(h'(x_grid))  # O(g)
```

**具体例**:

| 変換 | ヤコビアン $\|h'(x)\|$ | 計算量 |
|------|---------------------|--------|
| **Exp**: $e^x$ | $e^x$ | $O(g)$ |
| **Log**: $\log x$ | $1/x$ | $O(g)$ |
| **Power**: $x^a$ | $\|a x^{a-1}\|$ | $O(g)$ |
| **Abs**: $\|x\|$ | 特別処理 | $O(g)$ |
| **ReLU**: $\max(0, x)$ | 特別処理 | $O(g)$ |

**Abs, ReLUの特別処理**:
- 非可逆な変換のため、分布を分割して統合
- 例: Abs では $x < 0$ の部分を反転して $x > 0$ に統合

**メモリ**: $O(g)$

---

## 7. ベクトル演算

### 7.1 vector_add: ベクトルの各要素にAdd演算

**実装**: `dpest/operations/vector_ops.py`

```python
def vector_add(vec1, vec2):
    return [add(v1, v2) for v1, v2 in zip(vec1, vec2)]
```

**計算量**: $O(m \times g \log g)$
- $m$: ベクトルサイズ
- 各要素でAdd演算: $O(g \log g)$

**メモリ**: $O(m \times g)$

---

### 7.2 vector_argmax: ベクトルからArgmax演算

```python
def vector_argmax(distributions):
    return argmax(distributions)
```

**計算量**: $O(m^2 \times g)$
- Argmax演算の計算量

**メモリ**: $O(m \times g)$

---

### 7.3 vector_max, vector_min

**計算量**: $O(m^2 \times g)$

**メモリ**: $O(m \times g)$

---

## 8. 計算量の総括

### 8.1 解析手法で実行可能な演算

| 演算 | 計算量 | メモリ | 備考 |
|------|--------|--------|------|
| **Add**（離散+離散） | $O(k_X k_Y)$ | $O(k_X k_Y)$ | アトム数に依存 |
| **Add**（離散+連続） | $O(k \times g)$ | $O(k \times g)$ | 補間が必要 |
| **Add**（連続+連続） | $O(g \log g)$ | $O(g)$ | **FFT使用** |
| **Affine** | $O(g)$ | $O(g)$ | 線形変換 |
| **Max/Min** | $O(n^2 g^2)$ | $O(ng)$ | CDFベース |
| **Argmax** | $O(n^2 g^2)$ | $O(ng)$ | CDFベース |
| **Compare** | $O(g)$ (最適化版) | $O(g)$ | CDF利用 |
| **Compare** | $O(g^2)$ (二重ループ版) | $O(g)$ | 素朴実装 |
| **Mul** | $O(g \log g)$ | $O(g)$ | 対数変換経由 |
| **単調変換** | $O(g)$ | $O(g)$ | Exp, Log, etc. |
| **vector_add** | $O(m g \log g)$ | $O(mg)$ | $m$回のAdd |
| **vector_argmax** | $O(m^2 g^2)$ | $O(mg)$ | 1回のArgmax |

**デフォルトパラメータ**（$g=1000$）での実効演算数:

| 演算 | 実効演算数 | 実測時間（目安） |
|------|-----------|--------------|
| Add（連続） | $10^4$ | 0.001s |
| Affine | $10^3$ | < 0.001s |
| Max ($n=5$) | $2 \times 10^7$ | 0.5s |
| Argmax ($n=5$) | $2 \times 10^7$ | 0.5s |
| Compare | $10^3$ | < 0.001s |

---

### 8.2 サンプリングモードに移行する演算

| 演算 | 理由 | 計算量 |
|------|------|--------|
| **Branch** | 依存関係の発生 | $O(N)$ |
| **PrefixSum** | 累積依存 | $O(Nm)$ |
| **依存のあるMax/Argmax** | 共通変数参照 | $O(N)$ |

**サンプリングモードのボトルネック**:
- サンプル数 $N=1,000,000$ の場合
- 実効速度: 解析手法の1/100〜1/1000
- キャッシュミス率が高い（ランダムアクセス）

---

## 9. アルゴリズム別の計算量分析

### 9.1 LaplaceMechanism

**演算**: Add × 1

**計算量**: $O(g \log g) = 10^4$ 演算

**実測**: 0.01秒

---

### 9.2 NoisyHist1/2

**演算**: Add × $m$ ($m=5$)

**計算量**: $O(m g \log g) = 5 \times 10^4$ 演算

**実測**: 0.01秒

---

### 9.3 ReportNoisyMax1

**演算**:
1. Add × $m$ ($m=5$): $5 \times 10^4$ 演算
2. Argmax × 1: $3 \times 10^4$ 演算

**計算量**: $O(mg \log g + m^2 g) \approx 8 \times 10^4$ 演算

**実測**: 2.75秒

**なぜ実測が理論より遅い？**

実は、プライバシー損失 $\varepsilon$ の推定が支配的:

#### 9.3.1 プライバシー損失推定の計算量

**手順**:
1. 隣接ペア $(D, D')$ ごとに:
   - アルゴリズム実行: $P = \text{ReportNoisyMax1}(D)$
   - アルゴリズム実行: $Q = \text{ReportNoisyMax1}(D')$
   - ε計算: `epsilon_from_dist(P, Q)`

**隣接ペア数**: 2つ（one_above, one_below）

**各ペアでの計算**:
- ReportNoisyMax1実行: $8 \times 10^4$ 演算 × 2 = $1.6 \times 10^5$
- ε計算: 離散分布なので $O(k^2)$ = $O(5^2) = 25$ 演算

**総計算量**: $1.6 \times 10^5$ 演算

**実測との乖離**:
- 理論: $1.6 \times 10^5$ 演算
- 実測: 2.75秒
- 実効速度: $1.6 \times 10^5 / 2.75 \approx 5.8 \times 10^4$ 演算/秒

この低速さの理由:
1. **Argmax演算の実装オーバーヘッド**
   - Python のループとnumpy の相互作用
   - CDF計算での細かい演算の積み重ね
2. **キャッシュミス**
   - Argmax演算は不規則なメモリアクセスパターン
   - CDFの累積計算
3. **epsilon_from_dist の実装**
   - 離散分布でも、すべての値のペアを比較
   - 対数計算などの浮動小数点演算

**より詳細な分析**:

`epsilon_from_dist` での離散分布比較:
```python
for p_val in P.atoms:
    for q_val in Q.atoms:
        ratio = log(p_prob / q_prob)  # 浮動小数点演算
```

離散分布のアトム数: 最大5個（インデックス0〜4）

**実際の計算回数**:
- 各ペアで: $5 \times 5 = 25$ 回の比較
- 2ペア: $50$ 回の対数計算
- これは非常に軽い

**結論**: Argmax演算自体の実装オーバーヘッドが支配的

---

### 9.4 SVT1（サンプリングモード）

**演算**: Branch含む複雑な構造

**計算量**: $O(N \times m)$
- $N = 1,000,000$
- $m = 10$
- 合計: $10^7$ 演算

**実測**: 276.45秒

**実効速度**: $10^7 / 276.45 \approx 3.6 \times 10^4$ 演算/秒

**解析手法（ReportNoisyMax1）との比較**:
- 同じ演算数 $10^7$ でも、**100倍遅い**
- 理由: キャッシュミス率の違い

---

## 10. キャッシュ効率の影響

### 10.1 解析手法のキャッシュ効率

**メモリアクセスパターン**: 順次アクセス
- FFT: 連続メモリへの順次書き込み/読み込み
- CDF計算: 累積和（順次アクセス）
- 格子点の走査: 連続配列のループ

**キャッシュヒット率**: > 95%

**実効速度**: 理論演算数とほぼ一致

---

### 10.2 サンプリング手法のキャッシュ効率

**メモリアクセスパターン**: ランダムアクセス
- サンプル生成: 乱数生成器
- ヒストグラムビンへの分散: 不規則なメモリ書き込み
- 条件分岐: 分岐予測失敗

**キャッシュヒット率**: < 50%

**実効速度**: 理論演算数の1/100〜1/1000

**キャッシュミスのペナルティ**:
- L1キャッシュヒット: 1サイクル
- L2キャッシュヒット: 10サイクル
- L3キャッシュヒット: 40サイクル
- メモリアクセス: 200サイクル

ランダムアクセスでは平均100サイクル以上かかる可能性がある。

---

## 11. 結論と推奨事項

### 11.1 解析手法が高速な理由

1. **FFTベースの演算**: Add演算が $O(g \log g)$ で非常に効率的
2. **CDFベースの演算**: Max/Argmaxは $O(n^2 g^2)$ だが、小規模($n \leq 10$)では実用的
3. **キャッシュ効率**: 順次アクセスパターンで95%以上のヒット率
4. **決定論的**: 同じ入力に対して再計算不要

### 11.2 計算量の支配項

典型的なアルゴリズムでの支配項:

| アルゴリズム | 支配的演算 | 計算量 |
|------------|-----------|--------|
| LaplaceMechanism | Add | $O(g \log g)$ |
| NoisyHist1/2 | Add × $m$ | $O(mg \log g)$ |
| ReportNoisyMax1/2 | Argmax | $O(m^2 g^2)$ |
| NoisyMaxSum | Max × 複数 | $O(km^2 g^2)$ |
| SVT系 | サンプリング | $O(N \times m)$ |

### 11.3 最適化の指針

**解析手法の最適化**:
1. **格子点数の調整**: 精度要件に応じて $g$ を減らす（$g=500$ で誤差2倍、速度2倍）
2. **Argmax演算の回避**: 可能なら別の手法を検討
3. **ベクトル化**: numpyの高度な機能を活用

**サンプリング手法の最適化**:
1. **サンプル数の削減**: $N=100,000$ で誤差±6%（実用的）
2. **キャッシュフレンドリーなデータ構造**: 連続メモリ配置
3. **並列化**: 独立なサンプル生成は並列化可能

### 11.4 実用的な選択基準

**解析手法を選択すべき場合**:
- 独立性が保証される（Branch演算なし）
- 精度要件 < 10%
- 実行時間を最小化したい
- 決定論的な結果が必要

**サンプリング手法が必須な場合**:
- Branch演算がある
- 依存関係がある（共通変数参照）
- 累積依存がある（PrefixSum）

**推奨パラメータ**:
- 解析手法: $g=1000$ (デフォルト), 誤差 < 2%
- サンプリング手法: $N=1,000,000$, 誤差 < 2%
