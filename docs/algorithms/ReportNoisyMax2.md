# ReportNoisyMax2

## アルゴリズムの説明

ReportNoisyMax2は、ReportNoisyMax1の変種で、Laplaceノイズの代わりに指数分布ノイズを使用します。正しい実装であり、$\varepsilon$-差分プライバシーを満たします。

**出典**:
> Zeyu Ding, Yuxin Wang, Guanhong Wang, Danfeng Zhang, and Daniel Kifer. 2018.
> Detecting Violations of Differential Privacy. CCS 2018.
> Algorithm 6

**アルゴリズム**:
1. 入力ベクトル $q = (q_1, q_2, \ldots, q_m)$ を受け取る
2. 各要素 $i$ に対して、独立に指数分布ノイズ $\eta_i \sim \text{Exp}(2/\varepsilon)$ を追加
3. ノイズ付きベクトルの最大値を持つインデックスを返す: $\text{argmax}_i (q_i + \eta_i)$

**数式**:
$$
M(D) = \text{argmax}_i (q_i + \text{Exp}(2/\varepsilon))
$$

**プライバシー保証**: クエリの $L_\infty$ 感度が1の場合、$\varepsilon$-差分プライバシーを満たします。

## モード

**解析モード**

## プライバシー損失結果

| 項目 | 値 |
|------|-----|
| 入力サイズ | 5 |
| 推定 ε | 0.0964 |
| 理論 ε | 0.10 |
| 誤差 | -0.0036 (-3.6%) |
| 実行時間 | 2.75秒 |

**データソース**: `docs/privacy_loss_report.md`

## 理論的な計算量

### 解析モード

**全体計算量**: $O(m^2 \times g^2)$

ReportNoisyMax1と同じ計算量（Argmax演算が支配的）。

**実効計算量**（$m=5$, $g=1000$）: $\approx 2.5 \times 10^7$ 演算

**メモリ使用量**: $O(m \times g) = O(5000)$

## 理論的な誤差（精度）

ReportNoisyMax1と同様の誤差構造。総誤差 $O(10^{-2})$ が支配的。

## 理論と実験結果の比較分析

### 精度と速度

| 項目 | 値 | 評価 |
|------|-----|------|
| 相対誤差 | 3.6% | ReportNoisyMax1（6.9%）より良好 |
| 実行時間 | 2.75秒 | ReportNoisyMax1と同等 |

### 比較: DP-Sniper vs StatDP vs DPEST

| 手法 | 推定 ε | 実行時間 |
|------|--------|----------|
| DP-Sniper | 0.098 | 19秒 |
| StatDP | 0.100 | 48秒 |
| DPEST | 0.0964 | 2.75秒 |

**結論**: DPESTは7-17倍高速で、精度も同等。
