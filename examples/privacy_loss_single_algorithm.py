import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from dpest.analysis import estimate_algorithm
from dpest.algorithms import (
    noisy_hist1_dist,
    noisy_hist2_dist,
    report_noisy_max1_dist,
    report_noisy_max3_dist,
    report_noisy_max2_dist,
    report_noisy_max4_dist,
    laplace_vec_dist,
    laplace_parallel_dist,
    one_time_rappor_dist,
    rappor_dist,
    svt1_dist,
    svt2_dist,
    svt3_dist,
    svt4_dist,
    svt5_dist,
    svt6_dist,
    numerical_svt_dist,
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
    config = {"n_samples": 100_000, "hist_bins": 100}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("n_samples", "hist_bins"):
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
}


PairList = List[Tuple[np.ndarray, np.ndarray]]
AlgorithmHandler = Callable[[str, int, int, int], float]


def _default_pairs(n: int) -> PairList:
    return list(generate_patterns(n).values())


def _make_dist_handler(
    dist_func: Callable[[np.ndarray, float], Any],
    *,
    pairs_factory: Callable[[int], PairList] = _default_pairs,
) -> AlgorithmHandler:
    def handler(name: str, n: int, n_samples: int, hist_bins: int) -> float:
        pairs = pairs_factory(n)
        return estimate_algorithm(
            name,
            pairs,
            dist_func=dist_func,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )

    return handler


def _make_mechanism_handler(
    mechanism_factory: Callable[[], Any],
    *,
    pairs_factory: Callable[[int], PairList] = _default_pairs,
) -> AlgorithmHandler:
    def handler(name: str, n: int, n_samples: int, hist_bins: int) -> float:
        pairs = pairs_factory(n)
        mechanism = mechanism_factory()
        return estimate_algorithm(
            name,
            pairs,
            mechanism=mechanism,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )

    return handler


def _laplace_parallel_pairs(_: int) -> PairList:
    base_patterns = generate_patterns(INPUT_SIZES["LaplaceMechanism"])
    return [base_patterns["one_above"]]


def _laplace_parallel_dist(
    data: np.ndarray, eps: float
):  # pragma: no cover - thin wrapper
    return laplace_parallel_dist(data, 0.005, INPUT_SIZES["LaplaceParallel"])


_TRUNCATED_GEOMETRIC_PAIRS: PairList = [
    (np.array([2]), np.array([1])),
    (np.array([1]), np.array([0])),
]


def _truncated_pairs(_: int) -> PairList:
    return [(pair[0].copy(), pair[1].copy()) for pair in _TRUNCATED_GEOMETRIC_PAIRS]


ALGORITHM_HANDLERS: Dict[str, AlgorithmHandler] = {
    "NoisyHist1": _make_dist_handler(noisy_hist1_dist),
    "NoisyHist2": _make_dist_handler(noisy_hist2_dist),
    "ReportNoisyMax1": _make_dist_handler(report_noisy_max1_dist),
    "ReportNoisyMax2": _make_dist_handler(report_noisy_max2_dist),
    "ReportNoisyMax3": _make_dist_handler(report_noisy_max3_dist),
    "ReportNoisyMax4": _make_dist_handler(report_noisy_max4_dist),
    "LaplaceMechanism": _make_dist_handler(laplace_vec_dist),
    "LaplaceParallel": _make_dist_handler(
        _laplace_parallel_dist, pairs_factory=_laplace_parallel_pairs
    ),
    "SVT1": _make_dist_handler(svt1_dist),
    "SVT2": _make_dist_handler(svt2_dist),
    "SVT3": _make_dist_handler(svt3_dist),
    "SVT4": _make_dist_handler(svt4_dist),
    "SVT5": _make_dist_handler(svt5_dist),
    "SVT6": _make_dist_handler(svt6_dist),
    "SVT34Parallel": _make_mechanism_handler(lambda: SVT34Parallel(eps=0.1)),
    "NumericalSVT": _make_dist_handler(numerical_svt_dist),
    "PrefixSum": _make_mechanism_handler(lambda: PrefixSum(eps=0.1)),
    "OneTimeRAPPOR": _make_dist_handler(one_time_rappor_dist),
    "RAPPOR": _make_dist_handler(rappor_dist),
    "TruncatedGeometric": _make_mechanism_handler(
        lambda: TruncatedGeometricMechanism(eps=0.1, n=5),
        pairs_factory=_truncated_pairs,
    ),
}


def compute_epsilon(name: str, *, n_samples: int, hist_bins: int) -> float:
    """Estimate privacy loss for a single algorithm."""
    if name not in INPUT_SIZES:
        raise ValueError(f"Unknown algorithm: {name}")

    handler = ALGORITHM_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unsupported algorithm: {name}")

    return handler(name, INPUT_SIZES[name], n_samples, hist_bins)


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
    args = parser.parse_args()

    config = load_config(args.config)
    eps = compute_epsilon(
        args.algorithm,
        n_samples=config["n_samples"],
        hist_bins=config["hist_bins"],
    )
    ideal = IDEAL_EPS.get(args.algorithm)
    size = INPUT_SIZES.get(args.algorithm)
    if ideal is not None:
        ideal_disp = "∞" if math.isinf(ideal) else f"{ideal:.2f}"
        print(f"{args.algorithm} (n={size}): ε ≈ {eps:.4f} (ideal {ideal_disp})")
    else:
        print(f"{args.algorithm} (n={size}): ε ≈ {eps:.4f}")


if __name__ == "__main__":
    main()
