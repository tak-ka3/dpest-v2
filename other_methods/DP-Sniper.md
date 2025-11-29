# DP-Sniper: ブラックボックスによる差分プライバシー違反の発見

**出典**: Bichsel, B., Steffen, S., Bogunovic, I., & Vechev, M. (2021). DP-Sniper: Black-Box Discovery of Differential Privacy Violations using Classifiers. *IEEE S&P 2021*.

## 手法の概要

DP-Sniperは、機械学習の分類器を用いて差分プライバシー違反を自動的に発見する実用的なブラックボックス手法です。

### 主要な特徴

1. **ブラックボックスアプローチ**: アルゴリズムの内部構造を解析せず、入出力のみから違反を検出
2. **分類器ベース**: 機械学習分類器を訓練して、出力がどの入力から生成されたかを予測
3. **ほぼ最適な攻撃**: Neyman-Pearson補題に基づき、分類器を近似最適な攻撃に変換
4. **高い性能**: 最先端手法と比較して最大12.4倍強い保証、15.5倍高速
5. **浮動小数点脆弱性の検出**: ナイーブに実装されたアルゴリズムの浮動小数点演算の脆弱性を検出可能

### 中核となる2つのアイデア

**アイデア1: 分類器の訓練**
- 観測された出力が2つの可能な入力 $(a, a')$ のどちらから生成されたかを予測する分類器 $p_\theta(a|b)$ を訓練

**アイデア2: 攻撃への変換**
- 訓練した分類器を、差分プライバシーに対するほぼ最適な攻撃 $\mathcal{S}$ に変換
- 事後確率 $p(a|b)$ が閾値 $t$ を超える出力を選択するしきい値攻撃を構築

### 主要コンポーネント

1. **DP-Sniper（Algorithm 1）**: 与えられた入力ペア $(a, a')$ に対して近似最適な攻撃 $\mathcal{S}$ を発見
2. **DD-Search（Algorithm 2）**: DP-Sniperとヒューリスティックな入力探索を組み合わせて、最大のパワーを持つ証人 $(a, a', \mathcal{S})$ を発見

### 差分識別可能性（Differential Distinguishability）

**定義**: アルゴリズム $M: \mathbb{A} \to \mathbb{B}$ が $\xi$-差分識別可能（$\xi$-DD）であるとは、証人 $(a, a', S)$ が存在して：

$$\ln(\Pr[M(a) \in S]) - \ln(\Pr[M(a') \in S]) \geq \xi$$

**意味**: $M$ が $\xi$-DDならば、任意の $\varepsilon < \xi$ に対して $M$ は $\varepsilon$-DPを満たさない。つまり、$\xi$ は差分プライバシーの下界を与える。

## 手法の詳細を説明するにあたって必要な知識

### 1. 差分プライバシーの基礎

**$\varepsilon$-差分プライバシー**: メカニズム $M: \mathbb{A} \to \mathbb{B}$ が $\varepsilon$-DPを満たすとは、任意の隣接入力 $(a, a') \in \mathcal{N}$ と任意の事象 $S \in \mathcal{P}(\mathbb{B})$ について：

$$\ln(\Pr[M(a) \in S]) - \ln(\Pr[M(a') \in S]) \leq \varepsilon$$

- 小さい $\varepsilon$ ほど強いプライバシー保証
- $\varepsilon = 0$: 完全なプライバシー
- $\varepsilon = \infty$: プライバシー保証なし

**隣接入力（Neighborhood）**: $\mathcal{N} \subseteq \mathbb{A} \times \mathbb{A}$ は、1人のユーザーのデータのみが異なる入力ペアの集合

### 2. 攻撃のパワー（Power）

**定義**: 証人 $(a, a', S)$ のパワー $\mathcal{E}(a, a', S)$ は：

$$\mathcal{E}(a, a', S) := \ln(\Pr[M(a) \in S]) - \ln(\Pr[M(a') \in S])$$

**解釈**: パワーが大きいほど強力な違反の証拠となる。

**目標**: 最大のパワーを持つ攻撃を発見：
$$\arg\max_{S \in \mathcal{P}(\mathbb{B})} \mathcal{E}(a, a', S)$$

**課題**: $|\mathcal{P}(\mathbb{B})| = 2^{|\mathbb{B}|}$ であり、連続出力空間では実行不可能

### 3. 小さな確率の回避: c-パワー

**問題**: $\Pr[M(a) \in S]$ や $\Pr[M(a') \in S]$ が非常に小さいと、推定誤差が大きくなる

**解決策**: c-パワー $\mathcal{E}^{\geq c}(a, a', S)$ を導入：

$$\mathcal{E}^{\geq c}(a, a', S) := \ln^{\geq c}(\Pr[M(a) \in S]) - \ln^{\geq c}(\Pr[M(a') \in S])$$

where $\ln^{\geq c}(x) = \ln(\max(c, x))$

**利点**:
- 確率が $c$ 未満の場合、自動的に $c$ に切り上げ
- $c$ 未満の確率を推定する必要がなくなる
- 実験では $c = 0.01$ を使用

### 4. ランダム化攻撃

**決定的攻撃**: $S \in \mathcal{P}(\mathbb{B})$ - 各 $b \in \mathbb{B}$ について $b \in S$ か $b \notin S$ かを決定

**ランダム化攻撃**: $\mathcal{S} \in \mathbb{B} \to [0, 1]$ - 各 $b$ を確率 $\mathcal{S}(b) \in [0, 1]$ で含める

**利点**: ランダム化攻撃は決定的攻撃より強力な場合がある（特に小さな確率を避ける場合）

### 5. 事後確率（Posterior Probability）

**定義**: 出力 $b$ が与えられたとき、入力が $a$ である確率：

$$p(a|b) := \Pr[A = a \mid M(A) = b]$$

where $\Pr[A = a] = \Pr[A = a'] = \frac{1}{2}$ （一様事前分布を仮定）

**役割**: この確率が高い出力は、$a$ から生成された可能性が高いため、攻撃に含めるべき

### 6. しきい値攻撃（Threshold Attacks）

**定義**: パラメータ $t \in [0, 1]$ と $q \in [0, 1]$ に対して：

$$\Pr[b \in \mathcal{S}^{t,q}] = \Pr[p(a|b) \succ (t, q)] := [p(a|b) > t] + q \cdot [p(a|b) = t]$$

**直感**:
- 事後確率 $p(a|b)$ が閾値 $t$ より大きい出力を常に含める
- $p(a|b) = t$ の出力を確率 $q$ で含める

**重要性**: Neyman-Pearson補題により、しきい値攻撃を考慮するだけで十分

### 7. Neyman-Pearson補題

**内容**: 与えられた偽陽性率（false positive rate）の下で、真陽性率（true positive rate）を最大化する検定は尤度比検定（likelihood ratio test）である

**DP-Sniperへの適用**:
- $\Pr[M(a') \in \mathcal{S}] = c$ の制約下で $\Pr[M(a) \in \mathcal{S}]$ を最大化
- しきい値攻撃が最適解を与える

### 8. 分類器ファミリー

**ロジスティック回帰**:
$$p_{\theta,\theta_0}(a|b) = \sigma(b^T\theta + \theta_0)$$
where $\sigma(x) = 1/(1 + e^{-x})$

**ニューラルネットワーク**:
- 2つの隠れ層（サイズ10と5）
- ReLU活性化関数
- シグモイド出力層

## 手法の詳細

### 1. DP-Sniper: 攻撃の探索（Algorithm 1）

#### 全体フロー

```
DP-Sniper(M, a, a'):
  1. 分類器を訓練: θ ← TrainClassifier(M, a, a')
  2. M(a')から N サンプル生成: b'₀, ..., b'ₙ₋₁
  3. 各サンプルをスコアリング: p'ᵢ = pθ(a|b'ᵢ)
  4. スコアを降順ソート: p"₀, ..., p"ₙ₋₁
  5. 閾値選択: t† = p"_{⌊c·N⌋}
  6. タイブレーク選択: q† = (cN - Σ[p'ᵢ > t†]) / Σ[p'ᵢ = t†]
  7. 攻撃構築: S^{t†,q†} ← pθ(a|·) ⪰ (t†, q†)
  8. return S^{t†,q†}
```

#### ステップ1: 分類器の訓練

**目的**: 事後確率 $p(a|b)$ を近似する分類器 $p_\theta(a|b)$ を学習

**手順**:
1. $M(a)$ から $N_{\text{train}}$ サンプル生成
2. $M(a')$ から $N_{\text{train}}$ サンプル生成
3. データセット $\mathcal{D} = \{(a, b_i)\} \cup \{(a', b'_i)\}$ を準備
4. 二値分類問題として $p_\theta(a|b)$ を訓練

**訓練詳細**:
- ロジスティック回帰: 正則化付き確率的勾配降下法（10エポック、学習率0.3、モーメンタム0.3）
- ニューラルネットワーク: Adam最適化（10エポック、学習率0.1）

#### ステップ2-4: サンプリングとスコアリング

**目的**: $\Pr[M(a') \in \mathcal{S}^{t,q}] = c$ を満たすパラメータ $(t^\dagger, q^\dagger)$ を選択

**手順**:
1. $M(a')$ から新しい $N$ サンプルを生成: $b'_0, \ldots, b'_{N-1}$
2. 各サンプルに事後確率でスコア付け: $p'_i = p_\theta(a|b'_i)$
3. スコアを降順ソート: $p''_0 \geq p''_1 \geq \cdots \geq p''_{N-1}$

#### ステップ5-6: パラメータ選択

**目標**: 以下を満たす $(t^\dagger, q^\dagger)$ を選択：

$$\sum_{i=0}^{N-1} [p'_i > t^\dagger] + q^\dagger \sum_{i=0}^{N-1} [p'_i = t^\dagger] = cN$$

**閾値 $t^\dagger$ の選択**:
$$t^\dagger = p''_{\min(\lfloor c \cdot N \rfloor, N-1)}$$

これにより、高々 $c \cdot N$ 個のサンプルが $t^\dagger$ を超える

**タイブレーク $q^\dagger$ の選択**:
$$q^\dagger = \frac{cN - \sum_{i=0}^{N-1} [p'_i > t^\dagger]}{\sum_{i=0}^{N-1} [p'_i = t^\dagger]}$$

これにより、全体でちょうど $cN$ 個のサンプルがカバーされる

#### ステップ7: 攻撃の構築

**構築**: しきい値攻撃 $\mathcal{S}^{t^\dagger,q^\dagger}_\theta$ を以下のように定義：

$$\Pr[b \in \mathcal{S}^{t^\dagger,q^\dagger}_\theta] = [p_\theta(a|b) > t^\dagger] + q^\dagger \cdot [p_\theta(a|b) = t^\dagger]$$

#### サンプルサイズ $N$ の選択

**Guideline 1（経験的精度）**: 精度 $\omega$ と信頼度 $1-\alpha$ に対して：

$$N = \max\left\{\frac{2(1-c)}{\omega^2 \cdot c}, \frac{8(1-c)}{c}\left(\text{erf}^{-1}(1-2\alpha)\right)^2\right\}$$

**実験設定**: $c = 0.01$, $\omega = 0.005$, $\alpha = 0.05$ → $N = 10.7 \times 10^6$

### 2. 理論的保証（Theorem 2）

**定理2（近似最適攻撃）**: 任意の隣接入力 $(a, a') \in \mathcal{N}$ に対して、Algorithm 1が返す攻撃 $\mathcal{S}^{t^\dagger,q^\dagger}$ は以下を満たす（確率 $1-\alpha$ で）：

$$\mathcal{E}^{\geq c}(a, a', \mathcal{S}^{t^\dagger,q^\dagger}) \geq \max_{\mathcal{S} \in \mathbb{B} \to [0,1]} \mathcal{E}^{\geq c}(a, a', \mathcal{S}) - \frac{\rho}{c} - 2\left(\frac{\rho}{c}\right)^2$$

where $\rho = \sqrt{\frac{\ln(2/\alpha)}{2N}}$ かつ $\frac{\rho}{c} \leq \frac{1}{2}$

**意味**:
- DP-Sniperは最適なc-パワーに対して $\frac{\rho}{c} + 2(\frac{\rho}{c})^2$ の誤差内で近似
- 大きな $N$ を選ぶことで $\rho$ を小さくでき、近似精度が向上

#### 証明の概要（3ステップ）

**Step 1（Lemma 2）**: しきい値攻撃で十分

$$\max_{\mathcal{S} \in \mathbb{B} \to [0,1]} \mathcal{E}^{\geq c}(a, a', \mathcal{S}) \leq \max_{t,q \in [0,1]} \mathcal{E}^{\geq c}(a, a', \mathcal{S}^{t,q})$$

**Step 2（Lemma 4）**: 最適パラメータ $(t^*, q^*)$ は $\Pr[M(a') \in \mathcal{S}^{t^*,q^*}] = c$ を満たす

**Step 3（Lemma 5）**: $(t^\dagger, q^\dagger)$ は $(t^*, q^*)$ を近似（DKW不等式を使用）

### 3. DD-Search: 証人の探索（Algorithm 2）

#### 全体フロー

```
DD-Search(M):
  1. for (aᵢ, a'ᵢ) ∈ GenerateInputs() do:
       Sᵢ ← DP-Sniper(M, aᵢ, a'ᵢ)
  2. i* ← arg maxᵢ EstimateE(M, aᵢ, a'ᵢ, Sᵢ, N_check)
  3. return (aᵢ*, a'ᵢ*, Sᵢ*) and LowerBoundE(M, aᵢ*, a'ᵢ*, Sᵢ*, N_final)
```

#### 入力探索（GenerateInputs）

**StatDPパターンの使用**: 以下の7つの典型的なパターン（入力長5の例）：

| カテゴリ | $a$ | $a'$ |
|---------|-----|------|
| One Above | [1,1,1,1,1] | [2,1,1,1,1] |
| One Below | [1,1,1,1,1] | [0,1,1,1,1] |
| One Above Rest Below | [1,1,1,1,1] | [2,0,0,0,0] |
| One Below Rest Above | [1,1,1,1,1] | [0,2,2,2,2] |
| Half Half | [1,1,1,1,1] | [0,0,0,2,2] |
| All Above & All Below | [1,1,1,1,1] | [2,2,2,2,2] |
| X Shape | [1,1,0,0,0] | [0,0,1,1,1] |

**拡張**: DP-Sniperでは $(a, a')$ と $(a', a)$ の両方を含める（対称性を考慮）

#### パワーの推定（EstimateE）

**目的**: 各候補証人のパワーを推定して最良のものを選択

**方法**: $N_{\text{check}}$ サンプルを使用して推定：

$$\hat{\mathcal{E}}(a, a', \mathcal{S}) := \ln\left(\hat{P}^{N_{\text{check}}}_{M(a)\in\mathcal{S}}\right) - \ln\left(\hat{P}^{N_{\text{check}}}_{M(a')\in\mathcal{S}}\right)$$

#### 下界の計算（LowerBoundE）

**目的**: 統計的に健全な下界を提供（Theorem 1の保証）

**方法**: Clopper-Pearson区間を使用して $N_{\text{final}}$ サンプルで計算：

$$\underline{\mathcal{E}} = \ln\left(\underline{P}^{N_{\text{final}},\alpha/2}_{M(a)\in\mathcal{S}}\right) - \ln\left(\overline{P}^{N_{\text{final}},\alpha/2}_{M(a')\in\mathcal{S}}\right)$$

**保証（Theorem 1）**: 確率 $1-\alpha$ で $\underline{\mathcal{E}} \leq \mathcal{E}(a, a', \mathcal{S})$

**ハイパーパラメータ**:
- $N_{\text{check}} = N = 10.7 \times 10^6$
- $N_{\text{final}} = 2 \times 10^8$

### 4. 特徴変換（Feature Transformation）

#### 標準的な特徴変換

1. **特殊値のエンコーディング**: SVTの"abort"などの特殊な出力値をブール値フラグとして追加
2. **正規化**: すべての次元を同じ平均と経験的分散に正規化

#### 浮動小数点脆弱性の検出

**ビットパターン特徴**:
- 出力の64ビットIEEE 754浮動小数点表現を抽出
- 各ビットを入力層のニューロンに供給
- 例: 出力 $-1.5$ のビット表現 `10111111111110...0` を使用

**利点**: 浮動小数点演算の微妙な違いを自動的に検出

**実験結果**:
- 理論的に0.1-DPのはずのLaplace機構の実装が、実際には0.25-DPすら満たさないことを検出
- スナッピング機構では正しく0.098-DDのみを検出（修正版は正常）

### 5. 実験結果の概要

#### StatDPとの比較

**より強い保証**:
- 最大12.4倍強い差分識別可能性を検出
- 例: OneTimeRAPPOR（StatDP: 0.471, DP-Sniper: 0.600）

**高速化**:
- 平均15.5倍高速（ロジスティック回帰使用時）
- 例: LaplaceMechanism（StatDP: 47秒, DP-Sniper: 4分）

**新しいアルゴリズムへの汎化**:
- StatDPの評価に含まれないアルゴリズム（RAPPOR、PrefixSumなど）で最大7.9倍の改善
- StatDPは特定のアルゴリズムに過適合している可能性

#### 分類器の選択

**ロジスティック回帰 vs ニューラルネットワーク**:
- ほとんどの場合、同様の性能
- ニューラルネットワークはPrefixSumで有意に優れる
- 分類器ファミリーの選択は重要ではない

#### 検出例

**正しい実装**:
- NoisyHist1（理論0.1-DP）→ 0.098-DD検出（ほぼ一致）
- LaplaceMechanism（理論0.1-DP）→ 0.098-DD検出

**誤った実装**:
- ReportNoisyMax3（理論∞-DP）→ 0.249-DD検出
- SVT3（理論∞-DP）→ 0.182-DD検出

## StatDP・DPESTとの比較

### アプローチの根本的な違い

| 項目 | StatDP | DP-Sniper | DPEST |
|------|--------|-----------|-------|
| **手法分類** | 統計的仮説検定（Fisher's Exact Test） | 機械学習分類器 + Neyman-Pearson | 解析的/サンプリング確率分布計算 |
| **攻撃探索** | 事前定義パターン | 分類器学習 | - |
| **入力探索** | パターンヒューリスティック | パターンヒューリスティック | パターンヒューリスティック |
| **出力** | 違反検出（p値）+ 反例 | 違反検出（ξ）+ 反例 | ε推定値 + 誤差範囲 |
| **ブラックボックス** | セミ（ノイズなし実行が必要） | 完全 | 要コンパイル |

### 主要な利点と欠点

**DP-Sniperの強み**:
- ✅ 完全ブラックボックス（ノイズなし実行不要）
- ✅ 機械学習による適応的な攻撃探索
- ✅ StatDPより高速かつ強力な検出
- ✅ 浮動小数点脆弱性の自動検出
- ✅ 理論的最適性保証（c-power）

**DP-Sniperの制約**:
- ❌ 大量のサンプルが必要（$N \approx 10^7$）
- ❌ 定量的なε推定値を直接提供しない（ξの下界のみ）
- ❌ 分類器の訓練コスト

**StatDPとの違い**:
- StatDPは確率的変換 + 超幾何分布
- DP-Sniperは事後確率 + しきい値攻撃
- DP-Sniperは新しいアルゴリズムへの汎化が優れる

**DPESTとの違い**:
- DPESTはε値の定量推定に特化
- DP-Sniperは違反検出と反例生成に特化
- DP-Sniperは完全ブラックボックス、DPESTはホワイトボックス

### 使い分けの指針

- **DP-Sniperが適している場合**:
  - 完全ブラックボックステストが必要
  - 具体的な反例 $(a, a', S)$ が必要
  - 浮動小数点脆弱性の検出
  - ハッシュ関数など複雑な操作を含むアルゴリズム（RAPPOR等）
  - StatDPで対応できない新しいアルゴリズム

- **DPESTが適している場合**:
  - 定量的なε値の推定が必要
  - 解析モードでの超高速検証
  - 確率分布の詳細な解析

- **StatDPが適している場合**:
  - StatDPの評価に含まれる既知のアルゴリズム
  - より単純な統計的検定手法を好む場合

### まとめ

DP-Sniperは、機械学習を差分プライバシー検証に応用した革新的手法です。完全ブラックボックスでありながら、理論的に近似最適な攻撃を発見でき、従来手法より高速かつ強力な違反検出を実現しています。特に、浮動小数点脆弱性の自動検出は、実装の健全性を確認する上で非常に有用です。

一方、DPESTとは相補的な関係にあり、DP-Sniperは違反検出と反例生成に、DPESTは定量的ε推定に、それぞれ特化しています。用途に応じて使い分けることで、差分プライバシーアルゴリズムの包括的な検証が可能になります。
