import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
import numpy as np

from privacy_loss_report import (
    estimate_algorithm,
    generate_hist_pairs,
    generate_change_one_pairs,
    noisy_hist1_dist,
    noisy_hist2_dist,
    report_noisy_max1_dist,
    report_noisy_max3_dist,
    report_noisy_max2_dist,
    report_noisy_max4_dist,
    laplace_vec_dist,
    laplace_parallel_dist,
)
from dpest.mechanisms.sparse_vector_technique import (
    SparseVectorTechnique1,
    SparseVectorTechnique2,
    SparseVectorTechnique3,
    SparseVectorTechnique4,
    SparseVectorTechnique5,
    SparseVectorTechnique6,
    NumericalSVT,
)
from dpest.mechanisms.prefix_sum import PrefixSum
from dpest.mechanisms.rappor import OneTimeRappor, Rappor
from dpest.mechanisms.geometric import TruncatedGeometricMechanism
from dpest.mechanisms.parallel import SVT34Parallel


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


def compute_epsilon(name: str) -> float:
    """Estimate privacy loss for a single algorithm."""
    if name not in INPUT_SIZES:
        raise ValueError(f"Unknown algorithm: {name}")

    n = INPUT_SIZES[name]

    if name == "NoisyHist1":
        pairs = generate_hist_pairs(n)
        return estimate_algorithm(name, pairs, dist_func=noisy_hist1_dist)
    if name == "NoisyHist2":
        pairs = generate_hist_pairs(n)
        return estimate_algorithm(name, pairs, dist_func=noisy_hist2_dist)
    if name == "ReportNoisyMax1":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, dist_func=report_noisy_max1_dist)
    if name == "ReportNoisyMax3":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, dist_func=report_noisy_max3_dist)
    if name == "LaplaceMechanism":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, dist_func=laplace_vec_dist)
    if name == "LaplaceParallel":
        pairs = [generate_change_one_pairs(INPUT_SIZES["LaplaceMechanism"])[0]]
        dist = lambda data, eps: laplace_parallel_dist(data, 0.005, INPUT_SIZES["LaplaceParallel"])
        return estimate_algorithm(name, pairs, dist_func=dist)
    if name == "ReportNoisyMax2":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, dist_func=report_noisy_max2_dist)
    if name == "ReportNoisyMax4":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, dist_func=report_noisy_max4_dist)
    if name == "SVT1":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=SparseVectorTechnique1(eps=0.1))
    if name == "SVT2":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=SparseVectorTechnique2(eps=0.1))
    if name == "SVT3":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=SparseVectorTechnique3(eps=0.1))
    if name == "SVT4":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=SparseVectorTechnique4(eps=0.1))
    if name == "SVT5":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=SparseVectorTechnique5(eps=0.1))
    if name == "SVT6":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=SparseVectorTechnique6(eps=0.1))
    if name == "SVT34Parallel":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=SVT34Parallel(eps=0.1))
    if name == "NumericalSVT":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=NumericalSVT(eps=0.1))
    if name == "PrefixSum":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=PrefixSum(eps=0.1))
    if name == "OneTimeRAPPOR":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=OneTimeRappor())
    if name == "RAPPOR":
        pairs = generate_change_one_pairs(n)
        return estimate_algorithm(name, pairs, mechanism=Rappor())
    if name == "TruncatedGeometric":
        tg_pairs = [(np.array([2]), np.array([1])), (np.array([1]), np.array([0]))]
        return estimate_algorithm(
            name,
            tg_pairs,
            mechanism=TruncatedGeometricMechanism(eps=0.1, n=5),
        )

    raise ValueError(f"Unsupported algorithm: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate privacy loss for a single algorithm"
    )
    parser.add_argument("algorithm", help="Algorithm name")
    args = parser.parse_args()

    eps = compute_epsilon(args.algorithm)
    ideal = IDEAL_EPS.get(args.algorithm)
    size = INPUT_SIZES.get(args.algorithm)
    if ideal is not None:
        ideal_disp = "∞" if math.isinf(ideal) else f"{ideal:.2f}"
        print(f"{args.algorithm} (n={size}): ε ≈ {eps:.4f} (ideal {ideal_disp})")
    else:
        print(f"{args.algorithm} (n={size}): ε ≈ {eps:.4f}")


if __name__ == "__main__":
    main()
