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
) -> float:
    """Estimate privacy loss for an algorithm over given adjacency pairs."""

    eps_max = 0.0
    if n_samples is not None:
        set_sampling_samples(n_samples)

    if verbose:
        print(
            f"[estimate_algorithm] name={name}, "
            f"joint_dist_func={joint_dist_func}, dist_func={dist_func}, mechanism={mechanism}"
        )

    for D, Dp in pairs:
        if joint_dist_func is not None:
            args = (D, eps) if extra is None else (D, eps, *extra)
            P = joint_dist_func(*args)
            args_prime = (Dp, eps) if extra is None else (Dp, eps, *extra)
            Q = joint_dist_func(*args_prime)
            eps_val = epsilon_from_dist(P, Q)
        elif dist_func is not None:
            args = (D, eps) if extra is None else (D, eps, *extra)
            P = dist_func(*args)
            args_prime = (Dp, eps) if extra is None else (Dp, eps, *extra)
            Q = dist_func(*args_prime)
            if isinstance(P, list):
                eps_val = epsilon_from_list_joint(P, Q, bins=hist_bins)
            else:
                eps_val = epsilon_from_dist(P, Q)
        else:
            if mechanism is None:
                raise ValueError("mechanism or dist_func must be provided")
            P_samples = mechanism.m(D, n_samples)
            Q_samples = mechanism.m(Dp, n_samples)
            eps_val = epsilon_from_samples_matrix(P_samples, Q_samples, bins=hist_bins)

        eps_max = max(eps_max, eps_val)

    return eps_max


def generate_hist_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return adjacency pairs used for histogram-style mechanisms."""

    return list(generate_patterns(length).values())


def generate_change_one_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return standard change-one adjacency pairs."""

    return list(generate_patterns(length).values())
