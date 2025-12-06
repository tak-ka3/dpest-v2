"""Reusable privacy-loss estimation helpers."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core import Dist
from ..engine import set_sampling_samples
from ..utils.input_patterns import generate_patterns
from ..utils.privacy import (
    epsilon_from_dist,
    epsilon_from_list_joint,
    epsilon_from_samples_matrix,
)


def estimate_algorithm(
    name: str,
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    dist_func: Optional[Callable[..., Sequence[Dist] | Dist]] = None,
    joint_dist_func: Optional[Callable[..., Dist | List[Dist]]] = None,
    mechanism=None,
    eps: float = 0.1,
    n_samples: int = 100_000,
    extra: Optional[Iterable] = None,
    verbose: bool = False,
    hist_bins: int = 100,
    visualize_histogram: bool = False,
    return_mode: bool = False,
) -> float | tuple[float, str]:
    """Estimate privacy loss for an algorithm over given adjacency pairs."""
    print(f"[estimate_algorithm] Estimating privacy loss for {name}...")

    eps_max = 0.0
    if n_samples is not None:
        set_sampling_samples(n_samples)

    if verbose:
        print(
            f"[estimate_algorithm] name={name}, "
            f"joint_dist_func={joint_dist_func}, dist_func={dist_func}, mechanism={mechanism}"
        )

    mode_detected = None  # Track computation mode

    for pair_idx, (D, Dp) in enumerate(pairs):
        if visualize_histogram:
            print(f"\n{'='*70}")
            print(f"Processing pair {pair_idx + 1}/{len(pairs)}")
            print(f"{'='*70}")

        if joint_dist_func is not None:
            args = (D, eps) if extra is None else (D, eps, *extra)
            P = joint_dist_func(*args)
            args_prime = (Dp, eps) if extra is None else (Dp, eps, *extra)
            Q = joint_dist_func(*args_prime)
            eps_val = epsilon_from_dist(P, Q)
        elif dist_func is not None:
            args = (D, eps) if extra is None else (D, eps, *extra)
            P = dist_func(*args)
            if verbose:
                print("P:", P)  # Debug print
            args_prime = (Dp, eps) if extra is None else (Dp, eps, *extra)
            Q = dist_func(*args_prime)
            if verbose:
                print("Q:", Q)  # Debug print

            # Detect computation mode
            if pair_idx == 0:  # Only detect once
                if isinstance(P, list):
                    if len(P) > 0 and hasattr(P[0], '_joint_samples'):
                        mode_detected = "sampling"
                    else:
                        mode_detected = "analytic"
                else:
                    # Single output Dist
                    if hasattr(P, '_joint_samples'):
                        mode_detected = "sampling"
                    else:
                        mode_detected = "analytic"

            if isinstance(P, list):
                eps_val = epsilon_from_list_joint(P, Q, bins=hist_bins, verbose=visualize_histogram)
            else:
                eps_val = epsilon_from_dist(P, Q)
        else:
            if mechanism is None:
                raise ValueError("mechanism or dist_func must be provided")
            mode_detected = "mechanism_sampling"
            P_samples = mechanism.m(D, n_samples)
            Q_samples = mechanism.m(Dp, n_samples)
            eps_val = epsilon_from_samples_matrix(P_samples, Q_samples, bins=hist_bins, verbose=visualize_histogram)

        if visualize_histogram or verbose:
            print(f"\nPair {pair_idx + 1} epsilon: {eps_val:.4f}")

        # Early return on infinite privacy loss
        # Since ε = max(ε₁, ε₂, ..., εₙ), if any εᵢ = ∞, then ε = ∞
        # No need to evaluate remaining pairs
        import math
        if math.isinf(eps_val):
            print(f"[estimate_algorithm] Infinite privacy loss detected at pair {pair_idx + 1}/{len(pairs)}.")
            if len(pairs) > 1:
                print(f"[estimate_algorithm] Skipping remaining {len(pairs) - pair_idx - 1} pairs (result is guaranteed to be inf).")
            if mode_detected == "sampling":
                print(f"[estimate_algorithm] Computation mode: Sampling (Monte Carlo)")
            result = float("inf")
            if return_mode:
                return (result, mode_detected or "unknown")
            return result

        eps_max = max(eps_max, eps_val)

    # Print computation mode at the end
    if mode_detected == "sampling":
        print(f"[estimate_algorithm] Computation mode: Sampling (Monte Carlo)")

    if return_mode:
        return (eps_max, mode_detected or "analytic")
    return eps_max


def generate_hist_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return adjacency pairs used for histogram-style mechanisms."""

    return list(generate_patterns(length).values())


def generate_change_one_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return standard change-one adjacency pairs."""

    return list(generate_patterns(length).values())
