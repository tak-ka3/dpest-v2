# 依存関係解析の仕組み

このライブラリでは、計算グラフを走査して確率変数間の依存関係を検出し、
解析的に計算できるかサンプリングに切り替えるかを自動的に判断します。
ここではその流れを説明します。

## 依存関係の管理

- `core.Dist.__init__` が `sampler` を持つ分布に `_new_dep_id()` を通じて
  一意な ID を付与し、`dependencies` セットに保持します。
- 新しい乱数を生成する分布はこの ID を割り当て、既存の分布から派生する場合は
  入力の ID を和集合として引き継ぎます。
- 各 `Node` も同じ ID セットを持ち、計算グラフ全体で依存情報が伝播します。

## 計算グラフの走査

### 個別ノードの依存関係チェック

- `engine.Engine._analyze_node` が計算グラフを再帰的に走査し、
  ノードの入力の依存 ID を調べます。
- 子ノードがサンプリングを必要とする場合や、**単一ノードの入力間**で依存 ID に交差がある
  場合は `needs_sampling` フラグを立てます。
- フラグは上流から下流へ伝播し、最終的なノードがサンプリングを要するか
  どうかが決まります。

### リスト要素間の依存関係チェック（改善版）

**問題**: SVT5/SVT6のようなアルゴリズムでは、個別のノードには依存関係がなくても、
リストの異なる要素間で共通の確率変数（例: 共通のノイズ付き閾値T）に依存している場合があります。

**解決策** (`dpest/engine.py:131-143`):

```python
# リスト要素間の依存関係をチェック
all_deps = [getattr(r, 'dependencies', set()) for r in result]
for i in range(len(all_deps)):
    for j in range(i + 1, len(all_deps)):
        if all_deps[i] & all_deps[j]:  # 共通依存を検出
            needs_sampling_cross_deps = True
```

**具体例: SVT5**
```python
# 全クエリが同じノイズ付き閾値Tを共有
T = affine(Laplace(...), 1.0, t)  # dependencies = {id(lap_T)}

for Q in queries:
    over = geq(Q, T)  # dependencies = {id(Q), id(lap_T)}
    result.append(over)

# result[0].dependencies = {id(Q[0]), id(lap_T)}
# result[1].dependencies = {id(Q[1]), id(lap_T)}
# 共通部分 {id(lap_T)} を検出 → サンプリングモード
```

この改善により、`branch`演算を使わないアルゴリズムでも、共通の確率変数への依存が
正しく検出されるようになりました。

## 実行計画とフォールバック

- `engine.Engine._plan_execution` は解析結果に基づき、
  "analytic" または "sampling" の実行モードを選択します。
- 依存が無い場合は解析的なアルゴリズムを、依存がある場合は共同サンプルを
  生成して経験的な分布を構築します。
- `operations.Add.apply` や `operations.argmax_op.argmax_distribution` などの演算は、
  依存が検出されたときにサンプラーを用いて同時サンプルを生成し、
  結果の分布を推定します。

## まとめ

1. 分布生成時に依存 ID を割り振り、計算グラフに保持する。
2. `_analyze_node` で依存の有無とサンプリングの必要性を判定する。
3. 実行計画で解析的計算とサンプリングを自動的に切り替える。

この仕組みにより、複雑な依存関係を持つ差分プライバシーアルゴリズムでも
安全に扱うことができます。
