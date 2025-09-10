# 記法・共通準備

- 1 変量分布を `Dist = (atoms, density, support, err)` とし、`atoms = {(a_i, w_i)}`（点質量）、`density = {x: grid_x, f: grid_f, dx}`（連続部）とする。
- 誤差は`err = err_trunc + err_interp + err_quad`
- 連続同士の**和**や**畳み込み**は FFT を既定
- 混合では「**連続部**は連続の手順」「**点質量**は個別に写像/組合せ」し、最後に和集合（混合）として合成。

---

# 各Operation説明

## 1. Affine: `Z = aX + b`

### 数式（独立/依存に関わらず可）

- 連続部: $f_Z(z) = \frac{1}{|a|} f_X((z-b)/a)$
- 点質量: $(a_i, w_i) \mapsto (a\,a_i + b,\; w_i)$
- サポート: 区間写像。a=0 のときは**退化分布**（全質量が b へ）。

### 実装

1. 連続格子を逆写像して再サンプリング（必要に応じ `resample_to_grid`）。
    1. 先に写像先zの確率変数の値のグリッドを決めてから、逆写像を考えて、写像元xのグリッドを求めて、それぞれのグリッド値におけるPDF、点質量を補間等を用いて考える
2. 点質量は座標変換のみ。質量保存。
3. 誤差は補間+外挿成分（変換後の格子範囲が、元のサポート（確率質量がある範囲）より外にはみ出すことで外側は0と見做して切り捨てることで生じる誤差）を合算。

---

## 2. Add: `Z = X + Y`

### 数式（独立）

- 連続+連続: $f_Z = f_X * f_Y$（畳み込み）。
- 離散+離散: $P(Z=z) = \sum_{x+y=z} P(X=x)P(Y=y)$。
- 離散+連続: $f_Z(z) = \sum_i w_i\, f_Y(z-a_i)$。
    - $P(X=a_i​)=w_i​$

### 依存

- 依存がある場合は**同時密度** $f_{X,Y}$が必要: $f_Z(z) = \int f_{X,Y}(x, z-x)dx$。

### 実装

1. 連続同士は FFT。格子幅は `dx = dx_X ≈ dx_Y` に調整。
2. サポート（確率分布がゼロでない値域）は Minkowski 和。
    1. Minkowski和: A⊕B={a+b∣a∈A,b∈B} 
        1. e.g. A = [-10, 10], B = [-5, 5] , A⊕B = [-15, 15]

---

## 3. Mul: `Z = X · Y`

### 数式（独立）

- 一般形: $f_Z(z) = \int f_X(x) f_Y(z/x) \frac{1}{|x|}\,dx$。
    - 正値域（X,Y>0）では $\log Z = \log X + \log Y$ に写して**和へ** → 逆写像をとり、logZをZに変数変換して元に戻す。
- 離散×連続: $f_Z(z) = \sum_i w_i\, \tfrac{1}{|a_i|} f_Y(z/a_i)$。
- 離散×離散: $P(Z=z) = \sum_{i,j: a_i b_j = z} w_i v_j$。
    - $atoms_Z​= \{(a_i​b_j​,w_i​v_j​)\; \text{for all}\; i,j\}$

### 依存

- 同時密度 $f_{X,Y}$ を用いて $f_Z(z) = \iint \delta(z-xy) f_{X,Y}(x,y) dxdy$。

### 実装

1. 連続同士で、正値が事前に分かれば `log`ドメイン化が安定（加法→FFT→逆写像でヤコビアン）。

---

## 4. Div: `Z = X / Y`（任意）

- 独立: $f_Z(z) = \int f_X(zy) f_Y(y) |y| \, dy$。

---

## 5. Exp/Log/Power/Abs/ReLU（単調写像系）

### 変数変換（可逆区間ごと）

- 可逆写像 z=g(x) で $f_Z(z) = f_X(g^{-1}(z))\,\big|\tfrac{d}{dz} g^{-1}(z)\big|$。
- 点質量は像へ移送：$a_i \mapsto g(a_i)$
    - e.g. {a_1: p_1, a_2: p_2} → {g(a_1): p_1, g(a_2): p_2}

### 代表例

- `Exp`:  $g(x)=e^x,\; g^{-1}(z)=\log z,\; f_Z(z)=\tfrac{1}{z} f_X(\log z)$（z>0）
- `Log`（X>0）: $f_Z(z)= e^{z} f_X(e^{z})$
- `Power p>0`  **分割し、**x≥0・x<0 で逆像を合算
    - $f_Z​(z)=f_X​(+z^{1/p})⋅\frac{1}{p}​z^{\frac{1}{p}​−1}+f_X​(-z^{1/p})⋅\frac{1}{p}​z^{\frac{1}{p}​−1}$
- `Abs`: $f_{|X|}(z) = f_X(z)+f_X(-z),\; z\ge0$
- `ReLU`: $P(Z=0)=P(X\le 0),\; f_Z(z)=f_X(z)\mathbf{1}_{z>0}$
    - 1_{条件}で、条件を満たすと1を返し、条件を満たさないと0を返す関数（指示関数）

---

## 6. Max/Min

- `Max`: $F_{\max}(z) = \prod_{i=1}^k F_i(z)$、確率密度: $f_{\max}(z) = \sum_i f_i(z) \prod_{j\ne i} F_j(z)$
- `Min`: $1-F_{\min}(z) = \prod_{i}^k(1-F_i(z))$

### 依存

- 同時分布が必要: $F_{\max}(z)=P(\cap_i\{X_i\le z\})$
    - 多変量CDFを直接評価する必要がある $F_{X_1,\dots,X_n}(z,\dots,z)$

---

## 7. Argmax

- $P(\operatorname*{argmax}=i) = \int f_i(x) \prod_{j\ne i} F_j(x)\,dx.$

---

## 8. Branch/Case（条件付き混合）

- 条件 E によるブランチ： $P_Z = P_{Z|E}P(E) + P_{Z|\neg E}P(\neg E)$。
- `E` が閾値比較（例: X>c）なら $P(E)=1-F_X(c)$、`E` がX>Y なら独立で $P(E) = \int f_X(x)F_Y(x)dx$。

---

## 9. ノイズ機構

### Laplace(b)

- PDF: $f(z)=\tfrac{1}{2b}e^{-|z-\mu|/b}$

### Gaussian(σ)（任意）

- Pure-DP の解析では $\varepsilon=\infty$なので、Pure-DPしか対応していない時は扱わない

---

# 実装方針

```python
class Engine:
    """差分プライバシーε推定エンジンのメインクラス"""
    
    def __init__(self):
        self.operations = {
            'Laplace': Laplace,
            'Argmax': Argmax,
            'Add': Ad
        }
    # アルゴリズムの関数から計算グラフを作る
    def _build_computation_graph(algo_func):
    # 計算グラフの最適化（_build_computation_graphに処理を委譲する可能性もあり）
    def _optimize_graph(computation_graph):
        
    def compile(self, algo_func: Callable) -> Callable:
        """アルゴリズム関数から最適な計算グラフを生成して分布計算関数を返す（コンパイル）"""
        # 事前最適化：計算グラフの構築と最適化
	      computation_graph = self._build_computation_graph(algo_func)
	      optimized_graph = self._optimize_graph(computation_graph)
        
        def distribution_func(input_data):
            """入力データから出力分布を計算"""
            
            # 入力データから初期分布を作成
            input_dist = self._create_input_distribution(input_data)
            
            # 出力の確率分布を求める
            output_dist = self._execute_algorithm(optimized_graph, input_dist)
            
            return output_dist
            
        return distribution_func
        
def compile(algo_func):
	  """Compile algorithm to distribution function"""
	  engine = Engine()
	  return engine.compile(algo_func)
```

```python
@dataclass
class Interval:
    """区間を表現するクラス"""
    low: float
    high: float

class Dist:
    """確率分布の表現
    atoms: 点質量（アトム）のリスト [(value, weight), ...]
    density: 連続密度の格子近似
    support: サポート区間のリスト
    error_bounds: 誤差上界の情報
    """
    
    def __init__(self, 
                 atoms: Optional[List[Tuple[float, float]]] = None,
                 density: Optional[Dict] = None,
                 support: Optional[List[Interval]] = None,
                 error_bounds: Optional[Dict] = None):
        self.atoms = atoms or []  # [(value, weight), ...]
        self.density = density or {}  # grid-based density approximation
        self.support = support or []
        self.error_bounds = error_bounds or {}
```

```python
# 実際の使用例
@algo
def noisy_argmax_lap(x):
    """ラプラスノイズを加えたargmax"""
    z = x + Laplace(b=B, size=len(x))  # 各要素にラプラスノイズを追加
    return Argmax(z)

def main():
		all_datasets = [
				([1, 1, 1, 1, 1], [1, 1, 1, 1, 0]),
				([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),
				...
		]
		algo_compiled = compile(noisy_argmax_lap)
		for D, D_prime in all_datasets:
			# 下記の計算は確率変数の値の幅を揃えるために、P, Q = algo_compiled(D, D_prime)
			# のように同時に処理する可能性あり
			P = algo_compiled(D)        # D での出力分布
			Q = algo_compiled(D_prime)  # D' での出力分布
			result = estimate_eps(P, Q, cfg)
```