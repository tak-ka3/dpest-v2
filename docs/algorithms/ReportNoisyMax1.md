# ReportNoisyMax1

## アルゴリズムの説明

ReportNoisyMax1は、Report Noisy Max機構の正しい実装です。入力ベクトルの各要素にLaplaceノイズを追加し、最大値を持つインデックス（argmax）を返します。

**出典**:
> Zeyu Ding, Yuxin Wang, Guanhong Wang, Danfeng Zhang, and Daniel Kifer. 2018.
> Detecting Violations of Differential Privacy. CCS 2018.
> Algorithm 5

**アルゴリズム**:
1. 入力ベクトル $q = (q_1, q_2, \ldots, q_m)$ を受け取る
2. 各要素 $i$ に対して、独立にLaplaceノイズ $\eta_i \sim \text{Lap}(2/\varepsilon)$ を追加
3. ノイズ付きベクトルの最大値を持つインデックスを返す: $\text{argmax}_i (q_i + \eta_i)$ **数式**:

$$
M(D) = \text{argmax}_i (q_i + \text{Lap}(2/\varepsilon))
$$

**プライバシー保証**: クエリの $L_\infty$ 感度が1の場合、 $\varepsilon$ -DPを満たします。

**隣接性の定義**: $\|\cdot\|_\infty$ （L∞ノルム、任意の1要素の変更）
- 2つの入力が任意の1要素の値のみ異なる場合に隣接とみなす

## モード

**解析モード**

各要素に対してLaplace分布を格子近似し、Argmax演算を適用します。Argmax演算は累積分布関数（CDF）を使用して効率的に計算されます。

## プライバシー損失結果

| 項目 | 値 |
|------|-----|
| 入力サイズ | 5 |
| 推定 ε | 0.1069 |
| 理論 ε | 0.10 |
| 誤差 | +0.0069 (6.9%) |
| 実行時間 | 2.75秒 |

**データソース**: `docs/privacy_loss_report.md`

## 理論的な計算量

### 解析モード

**全体計算量**: $O(m^2 \times g^2)$ **内訳**:
1. **Laplace分布の生成**: $O(m \times g)$ - $m=5$ 個の独立なLaplace分布を生成
   - 各分布は格子点 $g=1000$ 上で評価
2. **Add演算（各要素に対する畳み込み）**: $O(m \times g \log g)$ - $m=5$ 回の独立な畳み込み
   - 各畳み込みは $O(g \log g)$ （FFTベース）
3. **Argmax演算**: $O(m^2 \times g^2)$ ← **支配的**
   - 外側ループ（各インデックス $i$ ）: $m=5$ 回
   - 内側ループ（他の分布 $j$ ）: $m-1=4$ 回
   - 各CDF計算（`_compute_cdf_on_grid`）: $O(g^2)$ - x_gridの各点（ $g$ 個）について、累積和をtrapzで計算（ $O(g)$ ）
     - 二重ループ構造: 外側 $g$ 回 × 内側 $O(g)$ = $O(g^2)$ **実効計算量**（ $m=5$ , $g=1000$ ）:

$$
m^2 \times g^2 = 25 \times 10^6 = 2.5 \times 10^7 \text{ 演算}
$$

**メモリ使用量**: $O(m \times g) = O(5000)$ **参照**: `docs/OPERATION_COMPLEXITY_ANALYSIS.md`
- Argmax演算（連続分布）: 3.1節

## 理論的な誤差（精度）

### 解析モードの誤差構造

Argmax演算は、各インデックスが最大となる確率を計算するため、誤差の構造が複雑です。

#### 1. 切断誤差（Truncation Error）

各Laplace分布（スケール $b = 2/\varepsilon = 20$ ）の切断誤差:

$$
\text{err}_{\text{trunc}} \approx e^{-\varepsilon R / 2} = e^{-0.1 \times 500 / 2} = e^{-25} \approx 10^{-11}
$$

（無視できる）

#### 2. CDF計算誤差

Argmax演算では、各インデックスが最大となる確率を計算するために、累積分布関数（CDF）を $m \times (m-1)$ 回計算します。各CDF計算には以下の誤差が含まれます：

- **台形則誤差**: $O(L^3/g^2) \approx O(10^{-3})$ （各CDF計算）
- **累積誤差**: $m \times (m-1) \times O(10^{-3}) = 20 \times O(10^{-3}) = O(10^{-2})$ #### 3. 補間誤差（Interpolation Error）

$$
\text{err}_{\text{interp}} = O(1/g^2) = O(10^{-6})
$$

**総誤差**: CDF計算の累積誤差 $O(10^{-2})$ が支配的です。

**実測誤差との一致**:
- 理論誤差: $O(10^{-2})$ - 実測誤差: $0.0069 \approx 7 \times 10^{-3}$ - 一致度: 良好

## 理論と実験結果の比較分析

### 精度の分析

| 項目 | 理論値 | 実測値 | 差分 |
|------|--------|--------|------|
| ε | 0.10 | 0.1069 | +0.0069 |
| 相対誤差 | - | 6.9% | - |

**評価**:
- 推定精度は **良好** だが、LaplaceMechanismやNoisyHist1（誤差0.2%）と比較すると **やや低い**
- 理由: Argmax演算が $m^2$ 回のCDF計算を必要とし、誤差が累積
- 理論誤差 $O(10^{-2})$ と実測誤差 $6.9\% = 0.0069$ は一致

### 実行時間の分析

| 項目 | 理論値 | 実測値 | 評価 |
|------|--------|--------|------|
| 計算量 | $O(m^2 \times g^2)$ | - | - |
| 演算数（ $m=5$ ） | $\approx 2.5 \times 10^7$ | - | - |
| 実行時間 | - | 2.75秒 | NoisyHist1の275倍 |

**評価**:
- **実行時間 2.75秒** は、NoisyHist1（0.01秒）と比較して **275倍遅い**
- 理由: Argmax演算の計算量が $O(m^2 \times g^2)$ と高く、 $g^2$ の項が支配的
  - NoisyHist1: $O(m \times g \log g) = 5 \times 10^4$ 演算 → 0.01秒
  - ReportNoisyMax1: $O(m^2 \times g^2) = 2.5 \times 10^7$ 演算 → 2.75秒
  - 演算数の比: $2.5 \times 10^7 / 5 \times 10^4 = 500$ 倍
  - 実行時間の比: $2.75 / 0.01 = 275$ 倍
- それでも、DP-Sniperの実行時間（22秒）と比較して **8倍高速**
- StatDPの実行時間（46秒）と比較して **17倍高速**

### 比較: DP-Sniper vs StatDP vs DPEST

| 手法 | 推定 ε | 実行時間 | 相対速度 |
|------|--------|----------|----------|
| DP-Sniper | 0.092 | 22秒 | 1x |
| StatDP | 0.092 | 46秒 | 0.48x |
| DPEST | 0.1069 | 2.75秒 | **8x** |

**結論**:
1. **精度**: DPESTは理論値に最も近い（誤差6.9%）
   - DP-Sniper: 誤差8% (0.10 → 0.092)
   - StatDP: 誤差8% (0.10 → 0.092)
   - DPEST: 誤差6.9% (0.10 → 0.1069)
2. **速度**: DPESTは他手法と比較して **8-17倍高速**
3. **Argmax演算のコスト**: $O(m^2 \times g^2)$ が支配的
   - $m$ が増加すると、実行時間は $m^2$ に比例して増加
   - 大規模な入力（ $m \gg 5$ ）では、実行時間が問題になる可能性

### 解析モードの課題

ReportNoisyMax1は、Argmax演算の計算コストが高いことを示します：

1. ** $O(m^2 \times g^2)$ の計算量**: $g=1000$ では $g^2 = 10^6$ が支配的
2. **誤差の累積**: $m^2$ 回のCDF計算により、誤差が累積
3. **改善の余地**:
   - CDF計算の最適化（trapzの代わりに累積和を使用）
   - 格子点数 $g$ の削減（精度とトレードオフ）
   - 近似Argmax（モンテカルロ近似など）

**参照**:
- `docs/OPERATION_COMPLEXITY_ANALYSIS.md` - Argmax演算の詳細分析（3.1節）
- `docs/ANALYTIC_MODE_SUPERIORITY.md` - 解析モードの優位性と課題
