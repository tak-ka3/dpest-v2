import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import numpy as np

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


def compute_epsilon(name: str, *, n_samples: int, hist_bins: int) -> float:
    """Estimate privacy loss for a single algorithm."""
    if name not in INPUT_SIZES:
        raise ValueError(f"Unknown algorithm: {name}")

    n = INPUT_SIZES[name]

    if name == "NoisyHist1":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=noisy_hist1_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "NoisyHist2":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=noisy_hist2_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "ReportNoisyMax1":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            dist_func=report_noisy_max1_dist,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "ReportNoisyMax3":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            dist_func=report_noisy_max3_dist,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "LaplaceMechanism":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=laplace_vec_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "LaplaceParallel":
        pairs = [generate_patterns(INPUT_SIZES["LaplaceMechanism"])["one_above"]]
        dist = lambda data, eps: laplace_parallel_dist(data, 0.005, INPUT_SIZES["LaplaceParallel"])
        return estimate_algorithm(
            name, pairs, dist_func=dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "ReportNoisyMax2":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            dist_func=report_noisy_max2_dist,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "ReportNoisyMax4":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            dist_func=report_noisy_max4_dist,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "SVT1":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=svt1_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "SVT2":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=svt2_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "SVT3":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=svt3_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "SVT4":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=svt4_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "SVT5":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=svt5_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "SVT6":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name, pairs, dist_func=svt6_dist, n_samples=n_samples, hist_bins=hist_bins
        )
    if name == "SVT34Parallel":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            mechanism=SVT34Parallel(eps=0.1),
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "NumericalSVT":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            dist_func=numerical_svt_dist,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "PrefixSum":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            mechanism=PrefixSum(eps=0.1),
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "OneTimeRAPPOR":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            dist_func=one_time_rappor_dist,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "RAPPOR":
        pairs = list(generate_patterns(n).values())
        return estimate_algorithm(
            name,
            pairs,
            dist_func=rappor_dist,
            n_samples=n_samples,
            hist_bins=hist_bins,
        )
    if name == "TruncatedGeometric":
        tg_pairs = [(np.array([2]), np.array([1])), (np.array([1]), np.array([0]))]
        return estimate_algorithm(
            name,
            tg_pairs,
            mechanism=TruncatedGeometricMechanism(eps=0.1, n=5),
            n_samples=n_samples,
            hist_bins=hist_bins,
        )

    raise ValueError(f"Unsupported algorithm: {name}")


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
