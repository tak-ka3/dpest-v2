# LaplaceParallel

## アルゴリズムの説明

LaplaceParallelは、単一のスカラー値に対して、複数の独立したLaplaceノイズを並列に追加するアルゴリズムです。並列合成（parallel composition）の例として使用されます。

**アルゴリズム**:
1. 入力スカラー値 $x$ を受け取る
2. $n$ 個の独立なLaplaceノイズ $\eta_i \sim \text{Lap}(1/\varepsilon)$ を生成
3. $n$ 個のノイズ付き値 $(x + \eta_1, x + \eta_2, \ldots, x + \eta_n)$ を出力

**数式**:
$$
M(D) = (x + \text{Lap}(1/\varepsilon), x + \text{Lap}(1/\varepsilon), \ldots, x + \text{Lap}(1/\varepsilon))
$$

**プライバシー保証**: 並列合成により、全体で $\varepsilon$-差分プライバシーを満たします。

## モード

**解析モード**

## プライバシー損失結果

| 項目 | 値 |
|------|-----|
| 入力サイズ | 20 |
| 推定 ε | 0.1010 |
| 理論 ε | 0.10 |
| 誤差 | +0.0010 (1.0%) |
| 実行時間 | 0.04秒 |

**データソース**: `docs/privacy_loss_report.md`

## 理論的な計算量

### 解析モード

**全体計算量**: $O(n \times g \log g)$

**内訳**:
1. **Laplace分布の生成**: $O(n \times g)$ - $n=20$ 個の独立なLaplace分布
2. **Add演算**: $O(n \times g \log g)$ - $n=20$ 回の独立な畳み込み

**実効計算量**（$n=20$, $g=1000$）: $20 \times 1000 \times 10 \approx 2 \times 10^5$ 演算

**メモリ使用量**: $O(n \times g) = O(20000)$

## 理論的な誤差（精度）

LaplaceMechanismと同様の誤差構造。総誤差 $O(10^{-3})$。

## 理論と実験結果の比較分析

### 精度と速度

| 項目 | 値 | 評価 |
|------|-----|------|
| 相対誤差 | 1.0% | 高精度 |
| 実行時間 | 0.04秒 | $n=20$ でもLaplaceMechanism（0.01秒）の4倍程度 |

### 比較: DP-Sniper vs StatDP vs DPEST

| 手法 | 推定 ε | 実行時間 | 相対速度 |
|------|--------|----------|----------|
| DP-Sniper | 0.035 | 60秒 | 1x |
| StatDP | 0.014 | 1560秒 | 0.04x |
| DPEST | 0.1010 | 0.04秒 | **1500x** |

**結論**:
1. **精度**: DPESTは理論値に最も近い（誤差1.0%）
   - DP-Sniper: 誤差65%（大きく過小評価）
   - StatDP: 誤差86%（大きく過小評価）
2. **速度**: DPEST は1500-39000倍高速
3. **スケーラビリティ**: $n$ が増加しても、DPESTは線形にスケール（$O(n \times g \log g)$）
