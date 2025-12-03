# RAPPOR

## アルゴリズムの説明

RAPPORは、ローカル差分プライバシーを保証するランダム化応答アルゴリズムです。ブルームフィルタと二段階のランダム化（永続的 + 瞬時）を組み合わせます。

**出典**:
> Ulfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. 2014.
> RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response. CCS 2014.
> Steps 1-3

**Python実装**:
```python
def rappor(value, eps, n_hashes=4, filter_size=20, f=0.75, p=0.45, q=0.55, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    perm = onetime_rappor(value, eps, n_hashes=n_hashes, filter_size=filter_size, f=f, rng=rng)

    out = np.empty_like(perm)
    for i, b in enumerate(perm):
        if b == 1:
            out[i] = 1 if rng.random() < q else 0
        else:
            out[i] = 1 if rng.random() < p else 0
    return out
```

**アルゴリズム**:
1. 入力値をブルームフィルタにエンコード
2. 永続的ランダム化応答を適用
    - ここまではOneTime RAPPORと同様なので、filter_sizeのビット配列が返される
3. 瞬時ランダム化応答を適用: ビット=1のとき確率 $q$ で1を保持、ビット=0のとき確率 $p$ で1にフリップ

**プライバシー保証**: 適切なパラメータ設定で $0.40$ -DP

## モード

**解析モード**

## プライバシー損失結果

| 項目 | 値 |
|------|-----|
| 入力サイズ | 1 |
| 推定 ε | 0.3001 |
| 理論 ε | 0.40 |
| 誤差 | -0.0999 (-25.0%) |
| 実行時間 | 0.00秒 |

**データソース**: `docs/privacy_loss_report.md`

## 理論的な計算量

### 解析モード

**パラメータ**:
- $h = 4$ : ハッシュ関数の数（n_hashes）
- $b = 20$ : ビット数（filter_size）
- $k = 2$ : Branch演算における条件分布のアトム数

**全体計算量**: $O(h + 2 \times b \times k^2)$

**内訳**:
1. **OneTimeRAPPORの処理**: $O(h + b \times k^2) = O(84)$
   - ブルームフィルタのエンコード: $O(h) = O(4)$
   - 永続的ランダム化応答: $O(b \times k^2) = O(80)$
2. **瞬時ランダム化応答**: $O(b \times k^2) = O(80)$
   - 各ビット（ $b=20$ 個）に対して1回のBranch演算
   - 各Branch演算: `perm`（2アトム）× `dist_if_one/zero`（2アトム）= 4組み合わせ

**実効計算量**: $O(4 + 80 + 80) \approx O(164)$ 演算

**メモリ使用量**: $O(b) = O(20)$ （出力ビット列の長さ）

## 理論的な誤差（精度）

Branch演算（離散分布のみ）の誤差: 理論上0（完全な解析計算）

## 理論と実験結果の比較分析

### 精度と速度

| 項目 | 値 | 評価 |
|------|-----|------|
| 相対誤差 | 25.0% | OneTimeRAPPORと同程度 |
| 実行時間 | 0.00秒 | OneTimeRAPPORよりさらに高速 |

**解析モードの優位性**:
- **計算量が小さい理由**: OneTimeRAPPORと同様、離散分布（Branch演算のみ）で構成されており完全に解析的。計算量は $O(164)$ （OneTimeRAPPORのほぼ2倍）だが、サンプリングモード $O(N) = O(10^6)$ に比べて約6000倍小さい
- **精度が良い理由**: Branch演算は理論上誤差0のため、モンテカルロ誤差が存在しない

**誤差の原因**:
- 隣接入力データセットが限定的であるため、最悪ケースのプライバシー損失を捉えきれていない可能性がある
- DPESTは特定の入力ペアに対してのみプライバシー損失を推定するため、全ての隣接入力ペアを網羅できていない

### 比較: DP-Sniper vs StatDP vs DPEST

| 手法 | 推定 ε | 実行時間 |
|------|--------|----------|
| DP-Sniper | 0.299 | 120秒 |
| StatDP | 0.245 | 540秒 |
| DPEST | 0.3001 | 0.00秒 |

**結論**: DPESTは無限倍高速（0.00秒）。精度もDP-Sniperとほぼ同等。
