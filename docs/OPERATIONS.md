# DPEST Operations

DPESTで使用可能な確率分布演算の一覧と計算量の説明。

## 基本演算

| オペレーション | 機能説明 | 数式 |
| --- | --- | --- |
| `add(x, y)` | 2つの確率変数の和 | $Z = X + Y$ |
| `affine(x, a, b)` | アフィン変換 | $Z = aX + b$ |
| `mul(x, y)` | 2つの確率変数の積 | $Z = X \times Y$ |
| `geq(x, y)` | 比較演算（離散分布を返す） | $C = \mathbb{1}_{X \geq Y}$ |
| `branch(c, t, f)` | 条件分岐による混合分布 | $Z = \begin{cases} T & \text{if } C=\text{True} \\ F & \text{if } C=\text{False} \end{cases}$ |
| `argmax(dists)` | 最大値のインデックス（離散） | $Z = \text{argmax}_i(X_1, \ldots, X_n)$ |
| `max_op(dists)` | 最大値そのもの | $Z = \max(X_1, \ldots, X_n)$ |
| `min_op(dists)` | 最小値 | $Z = \min(X_1, \ldots, X_n)$ |

## 使用例

### NoisyArgMaxアルゴリズム

```python
@auto_dist()
def report_noisy_max1(values: List[Dist], eps: float) -> Dist:
    """Adds Laplace noise (scale 2/eps) and returns argmax index distribution."""
    noise_dists = create_laplace_noise(b=2 / eps, size=len(values))
    return vector_argmax(vector_add(values, noise_dists))
```

### geqとbranchの使い方

```python
# 閾値を超える確率が70%の場合
cond = geq(noisy_value, threshold)  # cond = {True: 0.7, False: 0.3}

result = branch(cond, 100.0, 10.0)
# 結果: 70%の確率で100、30%の確率で10の混合分布
# atoms = [(100.0, 0.7), (10.0, 0.3)]
```

---

---

## 計算量（解析モード）

詳細な実装アルゴリズムについては [OPERATION_DETAILS.md](OPERATION_DETAILS.md) を参照してください。

パラメータ:
- $g$: 格子点数（デフォルト1000）
- $n$: 分布の個数
- $k$: アトム（点質量）の個数

| オペレーション | 数式 | 計算量 |
|--------------|------|--------|
| **add(x, y)** | $f_Z(z) = (f_X * f_Y)(z)$ | 連続×連続: $O(g \log g)$<br>連続×離散: $O(kg)$<br>離散×離散: $O(k_X k_Y)$ |
| **affine(x, a, b)** | $f_Z(z) = \frac{1}{\|a\|} f_X(\frac{z-b}{a})$ | 連続: $O(g)$<br>離散: $O(k)$ |
| **geq(x, y)** | $P(X \geq Y) = \int f_X(x) F_Y(x) dx$ | $O(g^2)$ |
| **branch(c, t, f)** | $f_Z = P(C) f_T + (1-P(C)) f_F$ | 連続: $O(g \log g)$<br>離散: $O(k)$ |
| **argmax(dists)** | $P(\text{argmax}=i) = \int f_i(x) \prod_{j \neq i} F_j(x) dx$ | 連続: $O(n^2 g^2)$<br>離散: $O(nk^n)$ |
| **max_op(dists)** | $f_{\max}(z) = \sum_i f_i(z) \prod_{j \neq i} F_j(z)$ | 連続: $O(n^2 g^2)$<br>離散: $O(k^n)$ |

### 計算量の補足説明

**add (連続×連続: O(g log g))**
- FFTベースの畳み込み演算を使用
- 両方の分布を共通格子に補間してから畳み込み

**add (連続×離散: O(kg))**
- k個の点質量それぞれに対して連続分布をシフト（各O(g)）
- 例: 定数（k=1）+ Laplace分布 = O(g)の単純なシフト操作

**geq (O(g²))**
- X, Yそれぞれについてg個の格子点
- 各Xの格子点xに対して、Y≤xとなる累積確率を計算（O(g)）
- 全体でg × g = O(g²)

**branch (連続: O(g log g))**
- 2つの分布（true/false）を共通格子に補間
- `scipy.interpolate.interp1d`がバイナリサーチを使用: O(log g) per point
- g個の点について補間: O(g log g)
- 補間後の加重平均: O(g)
- 支配的な項はO(g log g)

**argmax (連続: O(n² g²))**
- n個の分布について、各分布が最大となる確率を計算
- 各分布iについて: $\int f_i(x) \prod_{j \neq i} F_j(x) dx$
- 積分の計算: g個の格子点で評価（O(g)）
- 各格子点で(n-1)個のCDFを計算・乗算（O(ng)）
- 全体で n × g × g = O(n² g²)

**argmax (離散: O(nk^n))**
- n個の離散分布の全組み合わせを列挙（k^n通り）
- 各組み合わせで最大値のインデックスを判定（O(n)）
