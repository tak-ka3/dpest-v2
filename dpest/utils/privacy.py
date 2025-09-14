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
                if abs(p_val - q_val) < 1e-10:
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
    """Compute privacy loss for lists of distributions."""
    return sum(epsilon_from_dist(P, Q) for P, Q in zip(P_list, Q_list))


def epsilon_from_samples(P: np.ndarray, Q: np.ndarray, bins: int = 50) -> float:
    """Estimate ε from samples of two distributions."""
    unique = np.union1d(np.unique(P), np.unique(Q))
    if len(unique) <= bins:
        p_counts = np.array([np.mean(P == v) for v in unique])
        q_counts = np.array([np.mean(Q == v) for v in unique])
        ratios: List[float] = []
        for p, q in zip(p_counts, q_counts):
            if p > 0 and q > 0:
                ratios.append(p / q)
                ratios.append(q / p)
        if ratios:
            return float(np.log(max(ratios)))
        return float("inf")
    else:
        hist_range = (min(P.min(), Q.min()), max(P.max(), Q.max()))
        p_hist, _ = np.histogram(P, bins=bins, range=hist_range, density=True)
        q_hist, _ = np.histogram(Q, bins=bins, range=hist_range, density=True)
        ratios: List[float] = []
        for p_val, q_val in zip(p_hist, q_hist):
            if p_val > 1e-12 and q_val > 1e-12:
                ratios.append(p_val / q_val)
                ratios.append(q_val / p_val)
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
    unique = np.unique(combined, axis=0)

    if unique.shape[0] <= bins:
        p_counts = np.array([np.mean(np.all(P == u, axis=1)) for u in unique])
        q_counts = np.array([np.mean(np.all(Q == u, axis=1)) for u in unique])
        ratios: List[float] = []
        for p_val, q_val in zip(p_counts, q_counts):
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
