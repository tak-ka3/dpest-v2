import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import time
import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from dpest.analysis import estimate_algorithm
from dpest.algorithms import (
    noisy_hist1,
    noisy_hist2,
    report_noisy_max1,
    report_noisy_max3,
    report_noisy_max2,
    report_noisy_max4,
    laplace_vec,
    laplace_parallel,
    one_time_rappor,
    rappor,
    svt1,
    svt2,
    svt3,
    svt4,
    svt5,
    svt6,
    numerical_svt,
    noisy_max_sum,
    truncated_geometric,
)
from dpest.utils.input_patterns import generate_patterns
from dpest.mechanisms.prefix_sum import PrefixSum
from dpest.mechanisms.geometric import TruncatedGeometricMechanism
from dpest.mechanisms.parallel import SVT34Parallel


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "privacy_loss_single_config.json",
)


def load_config(path: str | None) -> dict:
    config = {"n_samples": 100_000, "hist_bins": 100, "seed": 42}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("n_samples", "hist_bins", "seed"):
            if key in data:
                config[key] = int(data[key])
    return config


INPUT_SIZES = {
    "LaplaceMechanism": 1,
    "LaplaceParallel": 20,
    "NoisyHist1": 5,
    "NoisyHist2": 5,
    "ReportNoisyMax1": 5,
    "ReportNoisyMax2": 5,
    "ReportNoisyMax3": 5,
    "ReportNoisyMax4": 5,
    "SVT1": 10,
    "SVT2": 10,
    "SVT3": 10,
    "SVT4": 10,
    "SVT5": 10,
    "SVT6": 10,
    "SVT34Parallel": 10,
    "NumericalSVT": 10,
    "PrefixSum": 10,
    "OneTimeRAPPOR": 1,
    "RAPPOR": 1,
    "TruncatedGeometric": 5,
    "TruncatedGeometricAlgo": 1,
    "NoisyMaxSum": 20,
}

IDEAL_EPS = {
    "LaplaceMechanism": 0.1,
    "LaplaceParallel": 0.1,
    "NoisyHist1": 0.1,
    "NoisyHist2": 10.0,
    "ReportNoisyMax1": 0.1,
    "ReportNoisyMax2": 0.1,
    "ReportNoisyMax3": float("inf"),
    "ReportNoisyMax4": float("inf"),
    "SVT1": 0.1,
    "SVT2": 0.1,
    "SVT3": float("inf"),
    "SVT4": 0.18,
    "SVT5": float("inf"),
    "SVT6": float("inf"),
    "SVT34Parallel": float("inf"),
    "NumericalSVT": 0.1,
    "PrefixSum": 0.1,
    "OneTimeRAPPOR": 0.8,
    "RAPPOR": 0.4,
    "TruncatedGeometric": 0.12,
    "TruncatedGeometricAlgo": 0.12,
    "NoisyMaxSum": float("inf"),
}


PairList = List[Tuple[np.ndarray, np.ndarray]]
AlgorithmHandler = Callable[[str, int, int, int], float]


def _resolve_dist_func(obj: Callable) -> Callable:
    dist_func = getattr(obj, "_dist_func", None)
    if dist_func is None:
        return obj
    return dist_func


def _default_pairs(n: int) -> PairList:
    return list(generate_patterns(n).values())


def _make_dist_handler(
    algo_or_dist: Callable[[np.ndarray, float], Any],
    *,
    pairs_factory: Callable[[int], PairList] = _default_pairs,
) -> AlgorithmHandler:
    dist_func = _resolve_dist_func(algo_or_dist)
    def handler(name: str, n: int, n_samples: int, hist_bins: int, visualize_histogram: bool = False) -> float:
        pairs = pairs_factory(n)
        return estimate_algorithm(
            name,
            pairs,
            dist_func=dist_func,
            n_samples=n_samples,
            hist_bins=hist_bins,
            visualize_histogram=visualize_histogram,
        )

    return handler


def _make_mechanism_handler(
    mechanism_factory: Callable[[], Any],
    *,
    pairs_factory: Callable[[int], PairList] = _default_pairs,
) -> AlgorithmHandler:
    def handler(name: str, n: int, n_samples: int, hist_bins: int, visualize_histogram: bool = False) -> float:
        pairs = pairs_factory(n)
        mechanism = mechanism_factory()
        return estimate_algorithm(
            name,
            pairs,
            mechanism=mechanism,
            n_samples=n_samples,
            hist_bins=hist_bins,
            visualize_histogram=visualize_histogram,
        )

    return handler


def _laplace_parallel_pairs(_: int) -> PairList:
    base_patterns = generate_patterns(INPUT_SIZES["LaplaceMechanism"])
    return [base_patterns["one_above"]]


def _laplace_parallel_dist(
    data: np.ndarray, eps: float
):  # pragma: no cover - thin wrapper
    dist_func = _resolve_dist_func(laplace_parallel)
    return dist_func(data, 0.005, INPUT_SIZES["LaplaceParallel"])


_TRUNCATED_GEOMETRIC_PAIRS: PairList = [
    (np.array([2]), np.array([1])),
    (np.array([1]), np.array([0])),
]


def _truncated_pairs(_: int) -> PairList:
    return [(pair[0].copy(), pair[1].copy()) for pair in _TRUNCATED_GEOMETRIC_PAIRS]


def _truncated_geometric_dist(data: np.ndarray, eps: float):  # pragma: no cover
    """Wrapper for truncated_geometric algorithm."""
    dist_func = _resolve_dist_func(truncated_geometric)
    return dist_func(data, eps=eps, n=INPUT_SIZES["TruncatedGeometricAlgo"])


ALGORITHM_HANDLERS: Dict[str, AlgorithmHandler] = {
    "NoisyHist1": _make_dist_handler(noisy_hist1),
    "NoisyHist2": _make_dist_handler(noisy_hist2),
    "ReportNoisyMax1": _make_dist_handler(report_noisy_max1),
    "ReportNoisyMax2": _make_dist_handler(report_noisy_max2),
    "ReportNoisyMax3": _make_dist_handler(report_noisy_max3),
    "ReportNoisyMax4": _make_dist_handler(report_noisy_max4),
    "LaplaceMechanism": _make_dist_handler(laplace_vec),
    "LaplaceParallel": _make_dist_handler(
        _laplace_parallel_dist, pairs_factory=_laplace_parallel_pairs
    ),
    "SVT1": _make_dist_handler(svt1),
    "SVT2": _make_dist_handler(svt2),
    "SVT3": _make_dist_handler(svt3),
    "SVT4": _make_dist_handler(svt4),
    "SVT5": _make_dist_handler(svt5),
    "SVT6": _make_dist_handler(svt6),
    "SVT34Parallel": _make_mechanism_handler(lambda: SVT34Parallel(eps=0.1)),
    "NumericalSVT": _make_dist_handler(numerical_svt),
    "PrefixSum": _make_mechanism_handler(lambda: PrefixSum(eps=0.1)),
    "OneTimeRAPPOR": _make_dist_handler(one_time_rappor),
    "RAPPOR": _make_dist_handler(rappor),
    "TruncatedGeometric": _make_mechanism_handler(
        lambda: TruncatedGeometricMechanism(eps=0.1, n=5),
        pairs_factory=_truncated_pairs,
    ),
    "TruncatedGeometricAlgo": _make_dist_handler(
        _truncated_geometric_dist, pairs_factory=_truncated_pairs
    ),
    "NoisyMaxSum": _make_dist_handler(noisy_max_sum),
}


def compute_epsilon(name: str, *, n_samples: int, hist_bins: int, visualize_histogram: bool = False) -> float:
    """Estimate privacy loss for a single algorithm."""
    if name not in INPUT_SIZES:
        raise ValueError(f"Unknown algorithm: {name}")

    handler = ALGORITHM_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unsupported algorithm: {name}")

    return handler(name, INPUT_SIZES[name], n_samples, hist_bins, visualize_histogram)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate privacy loss for a single algorithm"
    )
    parser.add_argument("algorithm", help="Algorithm name")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="JSON config path containing 'n_samples' and 'hist_bins'.",
    )
    parser.add_argument(
        "--visualize-histogram",
        action="store_true",
        help="Visualize histogram binning strategy and statistics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # シード値を設定して再現性を確保
    np.random.seed(args.seed)

    config = load_config(args.config)

    # 実行時間計測開始
    start_time = time.time()

    eps = compute_epsilon(
        args.algorithm,
        n_samples=config["n_samples"],
        hist_bins=config["hist_bins"],
        visualize_histogram=args.visualize_histogram,
    )

    # 実行時間計測終了
    elapsed_time = time.time() - start_time

    ideal = IDEAL_EPS.get(args.algorithm)
    size = INPUT_SIZES.get(args.algorithm)
    if ideal is not None:
        ideal_disp = "∞" if math.isinf(ideal) else f"{ideal:.2f}"
        print(f"\n{args.algorithm} (n={size}): ε ≈ {eps:.4f} (ideal {ideal_disp})")
    else:
        print(f"\n{args.algorithm} (n={size}): ε ≈ {eps:.4f}")

    # 実行時間を表示
    print(f"Execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
