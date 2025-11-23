"""Privacy-related utility functions.

This module provides helpers to estimate privacy loss between distributions
or sample sets.  The estimators were originally implemented in example
scripts but are now collected here for reuse across the project.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from ..core import Dist


def epsilon_from_dist(P: Dist, Q: Dist) -> float:
    """Compute privacy loss ``ε`` between two distributions."""
    # Check if distributions have joint samples (sampling mode)
    if (hasattr(P, '_joint_samples') and P._joint_samples is not None and
        hasattr(Q, '_joint_samples') and Q._joint_samples is not None):
        # Use sampling-based epsilon estimation
        P_samples = P._joint_samples[:, P._joint_samples_column] if hasattr(P, '_joint_samples_column') else P._joint_samples.flatten()
        Q_samples = Q._joint_samples[:, Q._joint_samples_column] if hasattr(Q, '_joint_samples_column') else Q._joint_samples.flatten()
        return epsilon_from_samples_matrix(
            P_samples.reshape(-1, 1),
            Q_samples.reshape(-1, 1),
            bins=100,
            verbose=False
        )

    if P.atoms and Q.atoms:
        max_ratio = 0.0
        for p_val, p_prob in P.atoms:
            if p_prob <= 0:
                continue
            q_prob = 0.0
            for q_val, q_p in Q.atoms:
                try:
                    numeric = (
                        isinstance(p_val, (int, float))
                        and isinstance(q_val, (int, float))
                    )
                    if numeric and abs(p_val - q_val) < 1e-10:
                        q_prob = q_p
                        break
                    if not numeric and p_val == q_val:
                        q_prob = q_p
                        break
                except TypeError:
                    if p_val == q_val:
                        q_prob = q_p
                        break
            if q_prob > 0:
                ratio = max(p_prob / q_prob, q_prob / p_prob)
                if ratio > max_ratio:
                    max_ratio = ratio
        return math.log(max_ratio) if max_ratio > 0 else float("inf")
    elif P.density and Q.density:
        # unify grid and compare densities
        p_x = P.density["x"]
        p_f = P.density["f"]
        q_x = Q.density["x"]
        q_f = Q.density["f"]
        min_x = min(p_x[0], q_x[0])
        max_x = max(p_x[-1], q_x[-1])
        unified_x = np.linspace(min_x, max_x, 2000)
        from scipy import interpolate

        p_interp = interpolate.interp1d(p_x, p_f, bounds_error=False, fill_value=1e-10)
        q_interp = interpolate.interp1d(q_x, q_f, bounds_error=False, fill_value=1e-10)
        p_unified = p_interp(unified_x)
        q_unified = q_interp(unified_x)
        ratios: List[float] = []
        for i in range(len(unified_x)):
            if p_unified[i] > 1e-10 and q_unified[i] > 1e-10:
                ratios.append(p_unified[i] / q_unified[i])
                ratios.append(q_unified[i] / p_unified[i])
        if ratios:
            return float(np.log(max(ratios)))
        return float("inf")
    else:
        return float("inf")


def estimate_privacy_loss(P: Dist, Q: Dist) -> float:
    """Estimate privacy loss ε between two ``Dist`` objects.

    This is an alias for :func:`epsilon_from_dist` kept for backwards
    compatibility.
    """

    return epsilon_from_dist(P, Q)


def epsilon_from_list(P_list: List[Dist], Q_list: List[Dist]) -> float:
    """Compute privacy loss for lists of distributions.

    This function uses marginal composition (sum of individual epsilons).
    For a more accurate estimate considering joint distribution, use
    epsilon_from_list_joint() instead.
    """
    return sum(epsilon_from_dist(P, Q) for P, Q in zip(P_list, Q_list))


def epsilon_from_list_joint(P_list: List[Dist], Q_list: List[Dist], bins: int = 100, verbose: bool = False) -> float:
    """Compute privacy loss using joint distribution of output vectors.

    This function is more accurate than epsilon_from_list() for algorithms
    with dependencies between outputs (e.g., SVT), as it considers the
    joint distribution rather than marginal distributions.

    Args:
        P_list: List of output distributions for dataset D
        Q_list: List of output distributions for dataset D'
        bins: Number of bins for histogram (if needed)
        verbose: If True, print histogram visualization

    Returns:
        Estimated epsilon using joint distribution
    """
    # Check if distributions have joint samples attached
    if (len(P_list) > 0 and hasattr(P_list[0], '_joint_samples') and
        len(Q_list) > 0 and hasattr(Q_list[0], '_joint_samples')):
        # Use the saved joint samples
        P_samples = P_list[0]._joint_samples
        Q_samples = Q_list[0]._joint_samples
        return epsilon_from_samples_matrix(P_samples, Q_samples, bins=bins, verbose=verbose)
    else:
        # Fallback to marginal composition
        import warnings
        warnings.warn(
            "Joint samples not available, falling back to marginal composition. "
            "For accurate joint distribution estimation, ensure the algorithm uses sampling mode.",
            UserWarning
        )
        return epsilon_from_list(P_list, Q_list)


def _value_mask(arr: np.ndarray, value: float) -> np.ndarray:
    if isinstance(value, float) and math.isnan(value):
        return np.isnan(arr)
    return arr == value


def epsilon_from_samples(P: np.ndarray, Q: np.ndarray, bins: int = 50) -> float:
    """Estimate ε from samples of two distributions."""
    unique = np.union1d(np.unique(P), np.unique(Q))
    ratios: List[float] = []

    if len(unique) <= bins:
        for v in unique:
            p_mask = _value_mask(P, v)
            q_mask = _value_mask(Q, v)
            p = np.mean(p_mask)
            q = np.mean(q_mask)
            if p > 0 and q > 0:
                ratios.append(p / q)
                ratios.append(q / p)
        if ratios:
            return float(np.log(max(ratios)))
        return float("inf")

    finite_P = P[~np.isnan(P)]
    finite_Q = Q[~np.isnan(Q)]
    p_nan = np.mean(np.isnan(P))
    q_nan = np.mean(np.isnan(Q))

    ratios: List[float] = []

    if finite_P.size > 0 and finite_Q.size > 0:
        hist_range = (min(finite_P.min(), finite_Q.min()), max(finite_P.max(), finite_Q.max()))
        p_hist, _ = np.histogram(finite_P, bins=bins, range=hist_range, density=True)
        q_hist, _ = np.histogram(finite_Q, bins=bins, range=hist_range, density=True)
        for p_val, q_val in zip(p_hist, q_hist):
            if p_val > 1e-12 and q_val > 1e-12:
                ratios.append(p_val / q_val)
                ratios.append(q_val / p_val)
    elif finite_P.size > 0 or finite_Q.size > 0:
        # 片方のみ有限値を持つ場合、共有質量がないため無限大
        return float("inf")

    if p_nan > 0 and q_nan > 0:
        ratios.append(p_nan / q_nan)
        ratios.append(q_nan / p_nan)
    elif p_nan > 0 or q_nan > 0:
        return float("inf")
    if ratios:
        return float(np.log(max(ratios)))
    return float("inf")


def create_mixed_histogram_bins(
    samples: np.ndarray,
    n_bins: int,
    discrete_threshold: int = 10
) -> tuple:
    """各次元のビニング関数を作成（整数・浮動小数点・NaN対応）

    Args:
        samples: (n_samples, n_dims) サンプル配列
        n_bins: 連続値（非整数）用のビン数
        discrete_threshold: 使用されない（互換性のため残す）

    Returns:
        (bin_functions, n_bins_per_dim): 各次元のビニング関数とビン数

    ビニング戦略:
        - NaN: bin 0
        - 整数値: 各整数に個別のビン（bin 1, 2, ...）
        - 非整数の浮動小数点値: 範囲を分割してヒストグラム化
    """
    n_dims = samples.shape[1]
    bin_functions = []
    n_bins_per_dim = []

    for dim in range(n_dims):
        col = samples[:, dim]

        # NaN以外の値を抽出
        non_nan = col[~np.isnan(col)]

        if len(non_nan) == 0:
            # 全てNaNの場合
            def bin_func_all_nan(x):
                return 0 if math.isnan(x) else -1
            bin_functions.append(bin_func_all_nan)
            n_bins_per_dim.append(1)
            continue

        # 整数値と非整数値を分離
        is_integer = np.mod(non_nan, 1) == 0
        integer_vals = non_nan[is_integer]
        float_vals = non_nan[~is_integer]

        unique_integers = np.unique(integer_vals) if len(integer_vals) > 0 else np.array([])
        has_floats = len(float_vals) > 0

        if not has_floats:
            # 整数値のみの場合（例: -1000, 0, 1など）
            discrete_vals = list(unique_integers)

            def make_discrete_binner(vals):
                def binner(x):
                    if math.isnan(x):
                        return 0  # NaN専用ビン
                    try:
                        return vals.index(x) + 1  # 整数値ビン (1-indexed)
                    except ValueError:
                        # 未知の整数値は最後のビンへ
                        return len(vals)
                return binner

            bin_functions.append(make_discrete_binner(discrete_vals))
            n_bins_per_dim.append(len(discrete_vals) + 1)  # NaN + 整数値

        elif len(unique_integers) == 0:
            # 非整数の浮動小数点値のみの場合（例: 3.521, 4.123など）
            min_val = float_vals.min()
            max_val = float_vals.max()

            # ビンの境界を計算
            if max_val == min_val:
                edges = [min_val - 0.5, min_val + 0.5]
            else:
                edges = np.linspace(min_val, max_val, n_bins + 1)

            def make_continuous_binner(edges_copy):
                def binner(x):
                    if math.isnan(x):
                        return 0  # NaN専用ビン
                    bin_id = np.digitize(x, edges_copy, right=False)
                    bin_id = max(1, min(bin_id, len(edges_copy)))
                    return bin_id  # 1-indexed (0 is for NaN)
                return binner

            bin_functions.append(make_continuous_binner(edges))
            n_bins_per_dim.append(n_bins + 1)  # NaN + 連続ビン

        else:
            # 整数値と非整数値の両方がある場合（混合）
            discrete_vals = list(unique_integers)
            min_val = float_vals.min()
            max_val = float_vals.max()

            # ビンの境界を計算
            if max_val == min_val:
                edges = [min_val - 0.5, min_val + 0.5]
            else:
                edges = np.linspace(min_val, max_val, n_bins + 1)

            num_discrete_bins = len(discrete_vals)

            def make_mixed_binner(int_vals, edges_copy, offset):
                def binner(x):
                    if math.isnan(x):
                        return 0  # NaN専用ビン
                    # 整数値チェック
                    if x % 1 == 0:
                        try:
                            return int_vals.index(x) + 1  # 整数ビン (1-indexed)
                        except ValueError:
                            # 未知の整数値は最後の整数ビンへ
                            return len(int_vals)
                    else:
                        # 非整数値は連続ビン
                        bin_id = np.digitize(x, edges_copy, right=False)
                        bin_id = max(1, min(bin_id, len(edges_copy)))
                        return offset + bin_id  # 整数ビンの後ろに配置
                return binner

            bin_functions.append(make_mixed_binner(discrete_vals, edges, num_discrete_bins))
            n_bins_per_dim.append(1 + num_discrete_bins + n_bins)  # NaN + 整数 + 連続

    return bin_functions, n_bins_per_dim


def samples_to_bin_ids(
    samples: np.ndarray,
    bin_functions: List
) -> np.ndarray:
    """サンプル配列をビンIDの配列に変換

    Args:
        samples: (n_samples, n_dims) サンプル配列
        bin_functions: 各次元のビニング関数

    Returns:
        bin_ids: (n_samples, n_dims) ビンIDの配列
    """
    n_samples, n_dims = samples.shape
    bin_ids = np.zeros((n_samples, n_dims), dtype=int)

    for dim in range(n_dims):
        for i in range(n_samples):
            bin_ids[i, dim] = bin_functions[dim](samples[i, dim])

    return bin_ids


def build_joint_histogram(
    bin_ids: np.ndarray,
    n_bins_per_dim: List[int] = None
) -> dict:
    """ビンIDから同時ヒストグラムを構築

    Args:
        bin_ids: (n_samples, n_dims) ビンID配列
        n_bins_per_dim: 各次元のビン数（現在は未使用、将来の拡張用）

    Returns:
        histogram: {tuple(bin_ids): count}
    """
    histogram = {}
    for row in bin_ids:
        key = tuple(row)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def epsilon_from_mixed_samples(
    P: np.ndarray,
    Q: np.ndarray,
    n_bins: int = 100,
    discrete_threshold: int = 10,
    verbose: bool = False
) -> float:
    """混合分布（離散値 + 連続値 + NaN）からε推定

    Args:
        P: データセットDの出力サンプル (n_samples, n_dims)
        Q: データセットD'の出力サンプル (n_samples, n_dims)
        n_bins: 連続値用のビン数
        discrete_threshold: この数以下のユニーク値なら離散として扱う
        verbose: Trueの場合、ヒストグラム情報を出力

    Returns:
        推定されたε値
    """
    # 1. 両方のデータを結合してビニング戦略を決定
    combined = np.vstack([P, Q])
    bin_functions, n_bins_per_dim = create_mixed_histogram_bins(
        combined, n_bins, discrete_threshold
    )

    if verbose:
        print("\n" + "=" * 70)
        print("Mixed Histogram Binning Visualization")
        print("=" * 70)
        print(f"\nTotal dimensions: {len(n_bins_per_dim)}")
        print(f"Bins per dimension: {n_bins_per_dim}")

        # 各次元の詳細を表示
        for dim in range(min(len(n_bins_per_dim), 10)):  # 最大10次元まで表示
            col = combined[:, dim]
            non_nan = col[~np.isnan(col)]
            is_integer = np.mod(non_nan, 1) == 0
            integer_vals = non_nan[is_integer]
            float_vals = non_nan[~is_integer]

    # 2. サンプルをビンIDに変換
    P_bins = samples_to_bin_ids(P, bin_functions)
    Q_bins = samples_to_bin_ids(Q, bin_functions)

    # 3. ヒストグラム構築
    P_hist = build_joint_histogram(P_bins, n_bins_per_dim)
    Q_hist = build_joint_histogram(Q_bins, n_bins_per_dim)

    if verbose:
        print("\n" + "=" * 70)
        print("Histogram Statistics")
        print("=" * 70)
        print(f"\nP (dataset D):")
        print(f"  Total samples: {len(P)}")
        print(f"  Unique patterns: {len(P_hist)}")

        print(f"\nQ (dataset D'):")
        print(f"  Total samples: {len(Q)}")
        print(f"  Unique patterns: {len(Q_hist)}")

        common_patterns = set(P_hist.keys()) & set(Q_hist.keys())
        print(f"\nCommon patterns: {len(common_patterns)}")

        # ビンID説明
        print(f"\nBin ID interpretation:")
        print(f"  bin 0: NaN")
        print(f"  bin 1+: Integer values (discrete) or continuous range bins")
        print(f"  Example: (42, 0, 0, ...) means Dim0=bin42, Dim1=NaN, Dim2=NaN, ...")

        # Top patterns
        print(f"\nTop 10 patterns in P (by count):")
        sorted_p = sorted(P_hist.items(), key=lambda x: x[1], reverse=True)
        for i, (pattern, count) in enumerate(sorted_p[:10]):
            prob = count / len(P)
            # パターンを短縮表示
            pattern_str = str(pattern) if len(pattern) <= 5 else f"{pattern[:5]}..."
            print(f"  {i+1:2d}. {pattern_str:40s} count={count:5d} prob={prob:.4f}")

        print(f"\nTop 10 patterns in Q (by count):")
        sorted_q = sorted(Q_hist.items(), key=lambda x: x[1], reverse=True)
        for i, (pattern, count) in enumerate(sorted_q[:10]):
            prob = count / len(Q)
            pattern_str = str(pattern) if len(pattern) <= 5 else f"{pattern[:5]}..."
            print(f"  {i+1:2d}. {pattern_str:40s} count={count:5d} prob={prob:.4f}")

    # 4. ε計算
    all_bins = set(P_hist.keys()) | set(Q_hist.keys())
    ratios: List[float] = []

    n_p = len(P)
    n_q = len(Q)

    for bin_key in all_bins:
        p_count = P_hist.get(bin_key, 0)
        q_count = Q_hist.get(bin_key, 0)
        p_prob = p_count / n_p
        q_prob = q_count / n_q

        # 片方が0、もう片方が非0の場合 → ε=∞
        if (p_prob > 0 and q_prob == 0) or (p_prob == 0 and q_prob > 0):
            if verbose:
                print("\n" + "=" * 70)
                print("Epsilon Calculation")
                print("=" * 70)
                print(f"\nExclusive pattern found:")
                print(f"  Pattern: {bin_key}")
                print(f"  P count: {p_count} (prob={p_prob:.6f})")
                print(f"  Q count: {q_count} (prob={q_prob:.6f})")
                print(f"\nThis pattern appears in one dataset but not the other.")
                print(f"Privacy loss: epsilon = inf")
                print("=" * 70 + "\n")
            return float("inf")

        if p_prob > 0 and q_prob > 0:
            ratios.append(p_prob / q_prob)
            ratios.append(q_prob / p_prob)

    if verbose:
        print("\n" + "=" * 70)
        print("Epsilon Calculation")
        print("=" * 70)
        if ratios:
            max_ratio = max(ratios)
            epsilon_val = np.log(max_ratio)
            print(f"\nMax probability ratio: {max_ratio:.4f}")
            print(f"Estimated epsilon: {epsilon_val:.4f}")
        else:
            print("\nNo common patterns found (epsilon = inf)")
        print("=" * 70 + "\n")

    if ratios:
        return float(np.log(max(ratios)))
    return float("inf")


def epsilon_from_samples_matrix(P: np.ndarray, Q: np.ndarray, bins: int = 100, verbose: bool = False) -> float:
    """Estimate ε from samples of vector-valued distributions.

    Supports mixed distributions with discrete values (including NaN) and continuous values.
    Uses adaptive histogram binning that treats discrete values as separate bins
    and continuous values with histogram discretization.

    Args:
        P: Samples from distribution P (n_samples, n_dims)
        Q: Samples from distribution Q (n_samples, n_dims)
        bins: Number of bins for continuous values
        verbose: If True, print histogram visualization

    Returns:
        Estimated epsilon value
    """
    P = np.asarray(P)
    Q = np.asarray(Q)

    if P.ndim == 1:
        return epsilon_from_samples(P, Q, bins)

    # 混合ヒストグラム法を使用（離散値・連続値・NaN対応）
    return epsilon_from_mixed_samples(P, Q, n_bins=bins, discrete_threshold=10, verbose=verbose)
