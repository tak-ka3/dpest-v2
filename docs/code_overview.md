# コードベース全体の概要

## プロジェクト構成
- `dpest/`: 差分プライバシー推定ライブラリ本体。`core.py` で確率分布 `Dist` と依存関係ノードを定義し、`engine.py` がアルゴリズムをコンパイルして解析 (解析モード) またはモンテカルロ (サンプリングモード) で実行する。
- `dpest/algorithms/`: SVT 系を含むアルゴリズム実装。`svt1.py` などの分布ベース実装と、examples から利用しやすい `*_dist` ラッパーを提供。
- `dpest/operations/`: 基本演算とサンプリング補助。`sample_op.py` の `Sampled` がサンプル列からヒストグラムや離散分布を構築する。
- `dpest/utils/`: 入力パターン生成 (`input_patterns.py`) とプライバシー損失評価 (`privacy.py`) を提供。
- `examples/`: 実行スクリプト。`privacy_loss_report.py` と `privacy_loss_single_algorithm.py` は JSON 設定ファイル (`privacy_loss_*_config.json`) からサンプル数 `n_samples` とヒストグラム分割数 `hist_bins` を読み込む。
- `tests/`: 追加された `test_svt1_joint_sampling.py` が SVT1 のサンプリング fallback が NaN 連鎖を保つことを検証する。

## エンジンとサンプリング fallback
- `dpest.engine.FallbackResult` でアルゴリズム側が「解析結果 (Dist またはそのリスト) とサンプラー」を返せるようにした。`engine.compile()` は最初の実行で依存解析を行い、サンプリングが必要と判断した場合 `_execute_sampling()` を通して fallback サンプラーを呼び出す。
- `_execute_sampling()` は `Sampled.from_samples()` を使ってジョイントサンプル配列から各要素の分布を構築し、同じサンプル配列を `_joint_samples` として各 `Dist` に保存する。これにより `epsilon_from_list_joint()` が真のジョイント分布を参照できる。

## SVT1 の挙動
- `dpest/algorithms/svt1.py` は `FallbackResult` を返し、 `_svt1_sampler()` でモンテカルロサンプルを生成する。カウンタ `count` が上限 `c` に達した時点で `aborted` を立て、以降の出力を `NaN` に固定することで「カウント超過後は常に NaN」という依存構造を守る。
- 以前観測された `[0, 1, 0, 1, 1, 1, ...]` のようなサンプルは、NaN を生成できず各要素を独立にサンプリングしていた旧実装が原因。現在はジョイントサンプラーを使うため、`NaN` 以降に 1 が続くことはない。

## Sampled.from_samples() の NaN 処理
- `Sampled._samples_to_dist()` では NaN の有無をチェックし、NaN を含む場合は `nan_mask` で切り分ける。非 NaN サンプルだけで分布 (離散またはヒストグラム) を構築し、NaN の割合 `nan_prob` を個別のアトムとして追加する。これは NaN が「一度発生したら後続すべて NaN」という SVT1 の挙動を表現するため。
- サンプル配列が多次元の場合、`from_samples()` が列ごとの `Dist` リストを生成するためのリスト内包表記が登場する。各列を独立に処理しつつ、同じジョイントサンプルを `_joint_samples` に共有する設計。

## プライバシー損失計算 (`dpest/utils/privacy.py`)
- `epsilon_from_samples_matrix()` は `P` と `Q` のサンプル行列からジョイント分布を推定する。`combined` から NaN を扱える独自の `unique_rows` を構築し、|unique_rows| ≤ `bins` の離散ケースでは各行列パターンごとに出現確率を計算する。
- 固有パターン数が `bins` を超えた場合は多次元ヒストグラム (`np.histogramdd`) を使い、`bins ** dim > 1_000_000` のようにビン数が爆発するケースでは安全のため各次元のマージナル ε を合計するルートへフォールバックする。

### `np.histogramdd` と `bins` の影響
- `np.histogramdd(P, bins=bins, range=ranges, density=True)` は `dim` 次元のジョイントヒストグラムを作る。各次元の範囲 `ranges` は P/Q の最小最大で揃えており、`density=True` により確率密度関数を近似する。
- ユニークパターン数が `bins` に縮退する場合 (たとえば SVT1 の {0,1,NaN}^10 で `c=2` のとき、およそ 56 パターン) は離散枝 (`unique_rows` 分岐) で正確な確率を計算する。この場合 `np.histogramdd` まで到達しない。
- `unique.shape[0] <= bins` が `True` になるかはサンプルの離散性次第。SVT1 の出力では上記の通り True になることが多い。False になった場合はジョイントヒストグラム枝に進み、計算量削減のため `bins ** dim > 1_000_000` 判定でマージナル合成へ切り替える。

## 設定ファイルと examples
- `examples/privacy_loss_report.py` と `examples/privacy_loss_single_algorithm.py` は `--config` で JSON を読み込み、`n_samples` と `hist_bins` を `estimate_algorithm()` に渡す。これによってサンプリング fallback も設定値を共有する (`set_sampling_samples()` を呼び出す)。
- レポート生成時は algorithms ディレクトリの `*_dist` 関数のみを利用し、examples 内での重複実装は削除済み。`svt*_joint_dist()` も撤去済みで、SVT 系ラッパーは `dpest/algorithms/wrappers.py` に集約されている。

## 疑問点への回答メモ
- `examples/privacy_loss_report.py` 内で SVT 系の関数は同ファイル末尾の `results.append(...)` で直接呼び出されており、他ファイルから参照されていない。
- `estimate_algorithm()` は `dpest/analysis/estimation.py` に移設済みで、ライブラリ全体から利用可能。examples だけに閉じている違和感は解消されている。
- SVT1 の出力配列は要素同士が依存しているため、ジョイント分布推定 (`epsilon_from_list_joint()` → `epsilon_from_samples_matrix()`) を使用する現行方針で問題ない。
- `FallbackResult` を用いた SVT1 サンプラーによって、サンプリング fallback でも NaN を含む配列サンプルを直接収集できるようになった。

## テスト状況
- `TMPDIR=/tmp venv/bin/python -m pytest` を実行すると、ホスト環境の制限で一時ディレクトリを作れず失敗する (`FileNotFoundError: No usable temporary directory`).
- `examples/privacy_loss_report.py` はレポートファイルを書き出そうとするが、リポジトリが読み取り専用のため `PermissionError: Operation not permitted` で終了する。書き込み可能な環境で再実行する必要がある。
