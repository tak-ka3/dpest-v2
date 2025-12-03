# SVT4 (Sparse Vector Technique 4)

## アルゴリズムの説明

SVT4は、SVT1の変種で、プライバシーパラメータの分割とノイズスケールが異なります。

**出典**: Lyu et al. 2017, Algorithm 4

**アルゴリズム**:
1. 閾値 $T = t + \text{Lap}(1/\varepsilon_1)$ を設定（ $\varepsilon_1 = \varepsilon/4$ ）
2. 各クエリ $q_i$ に対して：
   - ノイズ付きクエリ $\tilde{q}_i = q_i + \text{Lap}(1/\varepsilon_2)$ を計算（ $\varepsilon_2 = \varepsilon - \varepsilon_1$ ）
   - $\tilde{q}_i \geq T$ かを判定
   - TRUE の場合、カウンタをインクリメント
   - カウンタが $c$ に達したら、以降は NAN を出力

**数式**:

$$
T = t + \text{Lap}(4/\varepsilon), \quad \tilde{q}_i = q_i + \text{Lap}(4/(3\varepsilon))
$$

**プライバシー保証**: 理論的には $0.18$ -DP（ $\varepsilon=0.1$ のときの正確な値は約0.18）。

**隣接性の定義**: $\|\cdot\|_\infty$ （L∞ノルム）
- 2つのデータベース $D_1, D_2$ が隣接： $\max_i |D_1[i] - D_2[i]| \leq 1$ （各要素が最大1の変化）

## モード

**サンプリングモード**

## プライバシー損失結果

| 項目 | 値 |
|------|-----|
| 入力サイズ | 10 |
| 推定 ε | 0.1761 |
| 理論 ε | 0.18 |
| 誤差 | -0.0039 (-2.2%) |
| 実行時間 | 310.38秒 |

**データソース**: `docs/privacy_loss_report.md`

## 理論的な計算量

**サンプリングモード**: $O(N \times m) = O(10^7)$ 演算

## 理論的な誤差（精度）

**モンテカルロ誤差**: $O(1/\sqrt{N}) \approx 10^{-3}$ ## 理論と実験結果の比較分析

### 比較: DP-Sniper vs StatDP vs DPEST

| 手法 | 推定 ε | 実行時間 |
|------|--------|----------|
| DP-Sniper | 0.169 | 60秒 |
| StatDP | 0.170 | 240秒 |
| DPEST | 0.1761 | 310.38秒 |

**結論**: DPESTは理論値0.18に最も近い（誤差2.2%）。DP-Sniperより5.2倍遅いが、精度は最も高い。
