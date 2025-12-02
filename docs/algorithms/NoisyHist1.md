# NoisyHist1

## アルゴリズムの説明

NoisyHist1は、ヒストグラムの各ビンに独立したLaplaceノイズを追加する差分プライバシーアルゴリズムです。このアルゴリズムは正しい実装であり、$\varepsilon$-差分プライバシーを満たします。

**出典**:
> Zeyu Ding, Yuxin Wang, Guanhong Wang, Danfeng Zhang, and Daniel Kifer. 2018.
> Detecting Violations of Differential Privacy. CCS 2018.
> Algorithm 9

**アルゴリズム**:
1. 入力ヒストグラム $h = (h_1, h_2, \ldots, h_m)$ を受け取る
2. 各ビン $i$ に対して、独立にLaplaceノイズ $\eta_i \sim \text{Lap}(1/\varepsilon)$ を追加
3. ノイズ付きヒストグラム $(h_1 + \eta_1, h_2 + \eta_2, \ldots, h_m + \eta_m)$ を出力

**数式**:

$$
M(D) = (h_1 + \text{Lap}(1/\varepsilon), h_2 + \text{Lap}(1/\varepsilon), \ldots, h_m + \text{Lap}(1/\varepsilon))
$$

**プライバシー保証**: ヒストグラムの $L_1$ 感度が1の場合、$\varepsilon$-差分プライバシーを満たします。

## モード

**解析モード**

各ビンに対してLaplace分布を格子近似し、$m$ 個の独立な畳み込み演算を並列実行します。

## プライバシー損失結果

| 項目 | 値 |
|------|-----|
| 入力サイズ | 5 |
| 推定 ε | 0.1002 |
| 理論 ε | 0.10 |
| 誤差 | +0.0002 (0.2%) |
| 実行時間 | 0.01秒 |

**データソース**: `docs/privacy_loss_report.md`

## 理論的な計算量

### 解析モード

**全体計算量**: $O(m \times g \log g)$

**内訳**:
1. **Laplace分布の生成**: $O(m \times g)$
   - $m=5$ 個の独立なLaplace分布を生成
   - 各分布は格子点 $g=1000$ 上で評価
2. **Add演算（各ビンに対する畳み込み）**: $O(m \times g \log g)$
   - $m=5$ 回の独立な畳み込み
   - 各畳み込みは $O(g \log g)$（FFTベース）
3. **並列化**: 独立な演算のため、並列実行可能

**実効計算量**（$m=5$, $g=1000$）:

$$
5 \times 1000 \times \log_2(1000) \approx 5 \times 10^4 \text{ 演算}
$$

**メモリ使用量**: $O(m \times g) = O(5000)$

**参照**: `docs/OPERATION_COMPLEXITY_ANALYSIS.md`
- Add演算（連続+連続）: 1.1.3節

## 理論的な誤差（精度）

### 解析モードの誤差構造

**総誤差**: $\text{err}_{\text{total}} = \text{err}_{\text{trunc}} + \text{err}_{\text{interp}} + \text{err}_{\text{quad}}$

各ビンの誤差は独立であり、LaplaceMechanismと同様の誤差特性を持ちます。

#### 1. 切断誤差（Truncation Error）

$$
\text{err}_{\text{trunc}} \approx e^{-\varepsilon R} = e^{-0.1 \times 500} \approx 10^{-22}
$$

（無視できる）

#### 2. 補間誤差（Interpolation Error）

$$
\text{err}_{\text{interp}} = O(1/g^2) = O(10^{-6})
$$

#### 3. 数値積分誤差（Quadrature Error）

台形則による誤差（支配的）:

$$
\text{err}_{\text{quad}} = O(L^3/g^2) \approx O(10^{-3})
$$

**複数ビンの影響**: $m=5$ 個のビンがあっても、各ビンの誤差は独立なので、全体の誤差は単一ビンと同程度（$O(10^{-3})$）に保たれます。

## 理論と実験結果の比較分析

### 精度の分析

| 項目 | 理論値 | 実測値 | 差分 |
|------|--------|--------|------|
| ε | 0.10 | 0.1002 | +0.0002 |
| 相対誤差 | - | 0.2% | - |

**評価**:
- 推定精度は **非常に高く**、理論値との誤差は0.2%のみ
- $m=5$ 個のビンがあっても、精度は単一ビン（LaplaceMechanism）と同等
- これは各ビンの誤差が独立であり、適切に管理されていることを示す

### 実行時間の分析

| 項目 | 理論値 | 実測値 | 評価 |
|------|--------|--------|------|
| 計算量 | $O(m \times g \log g)$ | - | - |
| 演算数（$m=5$） | $\approx 5 \times 10^4$ | - | - |
| 実行時間 | - | 0.01秒 | 極めて高速 |

**評価**:
- **実行時間 0.01秒** は、$m=5$ 個の畳み込み演算を含むにもかかわらず極めて高速
- LaplaceMechanism（$m=1$）と同じ実行時間を達成
  - 理由: 各ビンの計算が独立 → 並列化 or FFTのオーバーヘッドが小さい
- DP-Sniperの実行時間（37秒）と比較して **3700倍高速**
- StatDPの実行時間（360秒）と比較して **36000倍高速**

### 比較: DP-Sniper vs StatDP vs DPEST

| 手法 | 推定 ε | 実行時間 | 相対速度 |
|------|--------|----------|----------|
| DP-Sniper | 0.098 | 37秒 | 1x |
| StatDP | 0.086 | 360秒 | 0.10x |
| DPEST | 0.1002 | 0.01秒 | **3700x** |

**結論**:
1. **精度**: DPESTは理論値に最も近い（誤差0.2%）
   - DP-Sniper: 誤差2% (0.10 → 0.098)
   - StatDP: 誤差14% (0.10 → 0.086)
   - DPEST: 誤差0.2% (0.10 → 0.1002)
2. **速度**: DPESTは他手法と比較して **3700倍以上高速**
3. **スケーラビリティ**: $m$ が増加しても、解析手法は $O(m \times g \log g)$ で線形にスケール

### 解析モードの優位性

NoisyHist1は、複数の独立な確率変数に対する解析手法の優位性を示します：

1. **並列化**: $m$ 個のビンの計算が独立 → 並列実行可能
2. **誤差制御**: 各ビンの誤差が独立 → 全体の誤差も制御可能
3. **FFT加速**: 各畳み込みを $O(g \log g)$ で計算
4. **サンプリング手法との比較**:
   - サンプリング: $O(N \times m)$ where $N=10^6$ → 実行時間360秒（StatDP）
   - 解析: $O(m \times g \log g)$ where $g=1000$ → 実行時間0.01秒

**参照**: `docs/ANALYTIC_MODE_SUPERIORITY.md` - 解析モードの理論的優位性の詳細分析
