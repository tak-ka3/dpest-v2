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


def epsilon_from_list_joint(P_list: List[Dist], Q_list: List[Dist], bins: int = 100) -> float:
    """Compute privacy loss using joint distribution of output vectors.

    This function is more accurate than epsilon_from_list() for algorithms
    with dependencies between outputs (e.g., SVT), as it considers the
    joint distribution rather than marginal distributions.

    Args:
        P_list: List of output distributions for dataset D
        Q_list: List of output distributions for dataset D'
        bins: Number of bins for histogram (if needed)

    Returns:
        Estimated epsilon using joint distribution
    """
    # Check if distributions have joint samples attached
    if (len(P_list) > 0 and hasattr(P_list[0], '_joint_samples') and
        len(Q_list) > 0 and hasattr(Q_list[0], '_joint_samples')):
        # Use the saved joint samples
        P_samples = P_list[0]._joint_samples
        Q_samples = Q_list[0]._joint_samples
        return epsilon_from_samples_matrix(P_samples, Q_samples, bins=bins)
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


def epsilon_from_samples_matrix(P: np.ndarray, Q: np.ndarray, bins: int = 100) -> float:
    """Estimate ε from samples of vector-valued distributions.

    Instead of summing privacy losses for each coordinate independently, this
    function treats each sample as a whole vector and constructs a joint
    histogram (multi-dimensional) over the vectors.  The privacy loss is then
    evaluated based on the probability (or density) of each bin in this joint
    histogram.
    """
    P = np.asarray(P)
    Q = np.asarray(Q)

    if P.ndim == 1:
        return epsilon_from_samples(P, Q, bins)

    combined = np.vstack([P, Q])
    has_nan = np.isnan(combined).any()

    unique_rows: List[np.ndarray] = []
    seen_patterns = set()
    print("Finding unique rows in combined samples...")
    for row in combined:
        mask = np.isnan(row)
        key = (tuple(mask.tolist()), tuple(np.where(mask, 0.0, row).tolist()))
        if key not in seen_patterns:
            seen_patterns.add(key)
            unique_rows.append(row)

    print("unique_rows found:", len(unique_rows))

    if has_nan or len(unique_rows) <= bins:
        ratios: List[float] = []
        for row in unique_rows:
            p_mask = np.all(
                np.where(np.isnan(row), np.isnan(P), P == row),
                axis=1,
            )
            q_mask = np.all(
                np.where(np.isnan(row), np.isnan(Q), Q == row),
                axis=1,
            )
            p_val = np.mean(p_mask)
            q_val = np.mean(q_mask)
            if p_val > 0 and q_val > 0:
                ratios.append(p_val / q_val)
                ratios.append(q_val / p_val)
        if ratios:
            return float(np.log(max(ratios)))
        return float("inf")

    dim = P.shape[1]
    if bins ** dim > 1_000_000:
        return float(sum(epsilon_from_samples(P[:, i], Q[:, i], bins) for i in range(dim)))

    ranges = [
        (min(P[:, d].min(), Q[:, d].min()), max(P[:, d].max(), Q[:, d].max()))
        for d in range(dim)
    ]
    p_hist, _ = np.histogramdd(P, bins=bins, range=ranges, density=True)
    q_hist, _ = np.histogramdd(Q, bins=bins, range=ranges, density=True)

    ratios: List[float] = []
    for p_val, q_val in zip(p_hist.ravel(), q_hist.ravel()):
        if p_val > 1e-12 and q_val > 1e-12:
            ratios.append(p_val / q_val)
            ratios.append(q_val / p_val)
    if ratios:
        return float(np.log(max(ratios)))
    return float("inf")
