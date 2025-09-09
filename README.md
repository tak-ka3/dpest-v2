# 確率分布演算 Operation の計算方法

本ライブラリで実装されている各Operationの確率分布計算方法について説明します。

---

## 1. Max演算

- **目的**: 複数の確率変数 $X_1, X_2, ..., X_k$ の最大値 $Z = \max(X_1, ..., X_k)$ の分布を計算します。
- **公式**:
    - 累積分布関数: $F_{\max}(z) = \prod_{i=1}^k F_i(z)$
    - 確率密度関数: $f_{\max}(z) = \sum_{i=1}^k f_i(z) \prod_{j\ne i} F_j(z)$
- **実装**:  
  - 離散分布のみの場合は全組合せを列挙し最大値ごとに集計。
  - 連続分布を含む場合は統一格子上で上記公式に従い数値計算します。

---

## 2. Min演算

- **目的**: 複数の確率変数 $X_1, ..., X_k$ の最小値 $Z = \min(X_1, ..., X_k)$ の分布を計算します。
- **公式**:
    - $1 - F_{\min}(z) = \prod_{i=1}^k (1 - F_i(z))$
    - 密度は $F_{\min}(z)$ の数値微分で得ます。
- **実装**:  
  - 離散分布のみの場合は全組合せを列挙し最小値ごとに集計。
  - 連続分布を含む場合は統一格子上で上記公式に従い数値計算します。

---

## 3. Add演算

- **目的**: $Z = X + Y$ の分布を計算します（$X, Y$ 独立）。
- **公式**:
    - 連続+連続: $f_Z(z) = (f_X * f_Y)(z)$（畳み込み, FFT利用）
    - 離散+離散: $P(Z=z) = \sum_{x+y=z} P(X=x)P(Y=y)$
    - 離散+連続: $f_Z(z) = \sum_i w_i f_Y(z-a_i)$
- **実装**:  
  - 入力の型（離散/連続）に応じて場合分けし、FFTや補間を用いて計算します。

---

## 4. Affine演算

- **目的**: $Z = aX + b$ の分布を計算します。
- **公式**:
    - 連続部: $f_Z(z) = \frac{1}{|a|} f_X\left(\frac{z-b}{a}\right)$
    - 点質量: $(a_i, w_i) \mapsto (a a_i + b, w_i)$
- **実装**:  
  - 連続部はグリッド変換とヤコビアン補正。
  - 点質量は座標変換のみ。

---

## 5. Argmax演算

- **目的**: $Z = \mathrm{argmax}_i X_i$ の分布（各インデックスが最大となる確率）を計算します。
- **公式**:
    - $P(\mathrm{argmax}=i) = \int f_i(x) \prod_{j\ne i} F_j(x) dx$
- **実装**:  
  - 離散分布のみの場合は全組合せを列挙し最大インデックスごとに集計。
  - 連続分布を含む場合は格子上で数値積分します。

---

各Operationの詳細な実装は [operations/max_op.py](operations/max_op.py), [operations/operations.py](operations/operations.py), [operations/argmax_op.py](operations/argmax_op.py) を参照してください。

## テスト

自動テストは `pytest` で実行できます:

```bash
pip install -r requirements.txt
pytest
```

手動で挙動を確認したい場合は、以下を実行してください:

```bash
python tests/test_operation.py
```

GitHub Actions でもプッシュやプルリクエスト時に加え、Actionsタブから手動でも同じテストを実行できます。
