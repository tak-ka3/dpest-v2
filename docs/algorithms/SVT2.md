# SVT2 (Sparse Vector Technique 2)

## アルゴリズムの説明

SVT2は、SVT1の変種で、TRUEが出力されるたびに閾値を再サンプリングします。

**出典**: Lyu et al. 2017, Algorithm 2

**アルゴリズム**:
1. 閾値 $T = t + \text{Lap}(c/\varepsilon_1)$ を設定（ $\varepsilon_1 = \varepsilon/2$ ）
2. 各クエリ $q_i$ に対して：
   - ノイズ付きクエリ $\tilde{q}_i = q_i + \text{Lap}(2c/\varepsilon_2)$ を計算（ $\varepsilon_2 = \varepsilon - \varepsilon_1$ ）
   - $\tilde{q}_i \geq T$ かを判定
   - TRUE の場合、カウンタをインクリメントし、**新しい閾値を再サンプリング**: $T = t + \text{Lap}(c/\varepsilon_1)$
   - カウンタが $c$ に達したら、以降は NAN を出力

**数式**:

$$
T_0 = t + \text{Lap}(c/\varepsilon_1), \quad \tilde{q}_i = q_i + \text{Lap}(2c/\varepsilon_2)
$$

$$
T_{i+1} = \begin{cases} t + \text{Lap}(c/\varepsilon_1) & \text{if } \tilde{q}_i \geq T_i \\ T_i & \text{otherwise} \end{cases}
$$

**プライバシー保証**: $\varepsilon$ -DP

**隣接性の定義**: $\|\cdot\|_\infty$ （L∞ノルム）
- 2つのデータベース $D_1, D_2$ が隣接： $\max_i |D_1[i] - D_2[i]| \leq 1$ （各要素が最大1の変化）

## モード

**サンプリングモード**

## プライバシー損失結果

| 項目 | 値 |
|------|-----|
| 入力サイズ | 10 |
| 推定 ε | 0.0843 |
| 理論 ε | 0.10 |
| 誤差 | -0.0157 (-15.7%) |
| 実行時間 | 282.88秒 |

**データソース**: `docs/privacy_loss_report.md`

## 理論的な計算量

**サンプリングモード**: $O(N \times m) = O(10^7)$ 演算

## 理論的な誤差（精度）

**モンテカルロ誤差**: $O(1/\sqrt{N}) \approx 10^{-3}$ 

## 理論と実験結果の比較分析

### 比較: DP-Sniper vs StatDP vs DPEST

| 手法 | 推定 ε | 実行時間 |
|------|--------|----------|
| DP-Sniper | 0.086 | 120秒 |
| StatDP | 0.032 | 180秒 |
| DPEST | 0.0843 | 282.88秒 |

**結論**: DPESTはDP-Sniperより2.4倍遅いが、StatDPと同等の精度。SVT1（276秒）と同様の実行時間。
