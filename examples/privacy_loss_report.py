import os
import sys
import math
import numpy as np
from typing import List, Tuple

# Allow importing the dpest package when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dpest.core import Dist
from dpest.engine import AlgorithmBuilder, vector_argmax, vector_max
from dpest.operations import add_distributions
from dpest.noise import create_laplace_noise, create_exponential_noise
from dpest.utils.input_patterns import generate_patterns

from dpest.mechanisms.noisy_hist import NoisyHist1, NoisyHist2
from dpest.mechanisms.report_noisy_max import (
    ReportNoisyMax1, ReportNoisyMax3,
)
from dpest.mechanisms.sparse_vector_technique import (
    SparseVectorTechnique1, SparseVectorTechnique2, SparseVectorTechnique3,
    SparseVectorTechnique4, SparseVectorTechnique5, SparseVectorTechnique6,
    NumericalSVT,
)
from dpest.mechanisms.laplace import LaplaceMechanism
from dpest.mechanisms.parallel import LaplaceParallel, SVT34Parallel
from dpest.mechanisms.prefix_sum import PrefixSum
from dpest.mechanisms.rappor import OneTimeRappor, Rappor
from dpest.mechanisms.geometric import TruncatedGeometricMechanism


# ---------------------------------------------------------------------------
# Utility functions for epsilon estimation
# ---------------------------------------------------------------------------

def epsilon_from_dist(P: Dist, Q: Dist) -> float:
    """Compute privacy loss ε between two distributions."""
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
        return np.log(max_ratio) if max_ratio > 0 else float('inf')
    elif P.density and Q.density:
        # unify grid and compare densities
        p_x = P.density['x']
        p_f = P.density['f']
        q_x = Q.density['x']
        q_f = Q.density['f']
        min_x = min(p_x[0], q_x[0])
        max_x = max(p_x[-1], q_x[-1])
        unified_x = np.linspace(min_x, max_x, 2000)
        from scipy import interpolate
        p_interp = interpolate.interp1d(p_x, p_f, bounds_error=False, fill_value=1e-10)
        q_interp = interpolate.interp1d(q_x, q_f, bounds_error=False, fill_value=1e-10)
        p_unified = p_interp(unified_x)
        q_unified = q_interp(unified_x)
        ratios = []
        for i in range(len(unified_x)):
            if p_unified[i] > 1e-10 and q_unified[i] > 1e-10:
                ratios.append(p_unified[i] / q_unified[i])
                ratios.append(q_unified[i] / p_unified[i])
        if ratios:
            return np.log(max(ratios))
        return float('inf')
    else:
        return float('inf')

def epsilon_from_list(P_list: List[Dist], Q_list: List[Dist]) -> float:
    return sum(epsilon_from_dist(P, Q) for P, Q in zip(P_list, Q_list))

def epsilon_from_samples(P: np.ndarray, Q: np.ndarray, bins: int = 50) -> float:
    """Estimate ε from samples of two distributions."""
    unique = np.union1d(np.unique(P), np.unique(Q))
    if len(unique) <= bins:
        p_counts = np.array([np.mean(P == v) for v in unique])
        q_counts = np.array([np.mean(Q == v) for v in unique])
        ratios = []
        for p, q in zip(p_counts, q_counts):
            if p > 0 and q > 0:
                ratios.append(p / q)
                ratios.append(q / p)
        if ratios:
            return np.log(max(ratios))
        return float('inf')
    else:
        hist_range = (min(P.min(), Q.min()), max(P.max(), Q.max()))
        p_hist, _ = np.histogram(P, bins=bins, range=hist_range, density=True)
        q_hist, _ = np.histogram(Q, bins=bins, range=hist_range, density=True)
        ratios = []
        for p, q in zip(p_hist, q_hist):
            if p > 1e-12 and q > 1e-12:
                ratios.append(p / q)
                ratios.append(q / p)
        if ratios:
            return np.log(max(ratios))
        return float('inf')

def epsilon_from_samples_matrix(P: np.ndarray, Q: np.ndarray, bins: int = 50) -> float:
    """Estimate ε from samples of vector-valued distributions.

    Instead of summing privacy losses for each coordinate independently,
    this function treats each sample as a whole vector and constructs a
    joint histogram (multi-dimensional) over the vectors. The privacy loss
    is then evaluated based on the probability (or density) of each bin in
    this joint histogram.

    Args:
        P: Samples from the first distribution, shape (n_samples, dim).
        Q: Samples from the second distribution, same shape as ``P``.
        bins: Number of bins per dimension for the histogram. If the number
            of unique vectors is less than or equal to ``bins``, the samples
            are treated as discrete and exact probabilities are computed.

    Returns:
        Estimated privacy loss ``ε``.
    """
    P = np.asarray(P)
    Q = np.asarray(Q)

    # Handle 1D case by delegating to scalar implementation
    if P.ndim == 1:
        return epsilon_from_samples(P, Q, bins)

    # Combine to inspect uniqueness of vectors
    combined = np.vstack([P, Q])
    unique = np.unique(combined, axis=0)

    # If vector outcomes are few, treat as discrete distribution
    if unique.shape[0] <= bins:
        p_counts = np.array([np.mean(np.all(P == u, axis=1)) for u in unique])
        q_counts = np.array([np.mean(np.all(Q == u, axis=1)) for u in unique])
        ratios = []
        for p_val, q_val in zip(p_counts, q_counts):
            if p_val > 0 and q_val > 0:
                ratios.append(p_val / q_val)
                ratios.append(q_val / p_val)
        if ratios:
            return float(np.log(max(ratios)))
        return float('inf')

    # Otherwise approximate with a multi-dimensional histogram for densities
    dim = P.shape[1]
    # Fall back to coordinate-wise estimation if the joint histogram would be
    # too large (bins^dim bins)
    # ベクトル全体を直接推定できない場合でも、各次元に分解して逐次合成を適用すれば、
    # 差分プライバシーの安全な（過小評価にはならない）推定になる
    if bins ** dim > 1_000_000:
        return float(
            sum(epsilon_from_samples(P[:, i], Q[:, i], bins) for i in range(dim))
        )

    ranges = []
    for d in range(dim):
        ranges.append(
            (
                min(P[:, d].min(), Q[:, d].min()),
                max(P[:, d].max(), Q[:, d].max()),
            )
        )

    p_hist, _ = np.histogramdd(P, bins=bins, range=ranges, density=True)
    q_hist, _ = np.histogramdd(Q, bins=bins, range=ranges, density=True)

    ratios = []
    for p_val, q_val in zip(p_hist.ravel(), q_hist.ravel()):
        if p_val > 1e-12 and q_val > 1e-12:
            ratios.append(p_val / q_val)
            ratios.append(q_val / p_val)
    if ratios:
        return float(np.log(max(ratios)))
    return float('inf')

# ---------------------------------------------------------------------------
# Analytic implementations using operations
# ---------------------------------------------------------------------------

def noisy_hist1_dist(a: np.ndarray, eps: float) -> List[Dist]:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=1/eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)

def noisy_hist2_dist(a: np.ndarray, eps: float) -> List[Dist]:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)

def report_noisy_max1_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=2/eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    return vector_argmax(z_dists)

def report_noisy_max3_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=2/eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    return vector_max(z_dists)

def report_noisy_max2_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_exponential_noise(b=2/eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    dist = vector_argmax(z_dists)
    dist.normalize()
    return dist

def report_noisy_max4_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_exponential_noise(b=2/eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    dist = vector_max(z_dists)
    dist.normalize()
    return dist

def laplace_vec_dist(a: np.ndarray, eps: float) -> List[Dist]:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=1/eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)

def laplace_parallel_dist(a: np.ndarray, eps_each: float, n_parallel: int) -> List[Dist]:
    x_dist = Dist.deterministic(float(a.item(0)))
    noise_list = create_laplace_noise(b=1/eps_each, size=n_parallel)
    return [add_distributions(x_dist, n) for n in noise_list]

# ---------------------------------------------------------------------------
# Estimation driver
# ---------------------------------------------------------------------------

def estimate_algorithm(name: str, pairs: List[Tuple[np.ndarray, np.ndarray]], *,
                       dist_func=None, mechanism=None, eps: float = 0.1,
                       n_samples: int = 10000, extra=None) -> float:
    eps_max = 0.0
    for D, Dp in pairs:
        if dist_func is not None:
            if extra is None:
                P = dist_func(D, eps)
                Q = dist_func(Dp, eps)
            else:
                P = dist_func(D, eps, *extra)
                Q = dist_func(Dp, eps, *extra)
            if isinstance(P, list):
                eps_val = epsilon_from_list(P, Q)
            else:
                eps_val = epsilon_from_dist(P, Q)
        else:
            P_samples = mechanism.m(D, n_samples)
            Q_samples = mechanism.m(Dp, n_samples)
            eps_val = epsilon_from_samples_matrix(P_samples, Q_samples)
        if eps_val > eps_max:
            eps_max = eps_val
    return eps_max

def generate_hist_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the standard set of adjacency pairs of given length."""
    return list(generate_patterns(length).values())


def generate_change_one_pairs(length: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return the standard set of adjacency pairs of given length."""
    return list(generate_patterns(length).values())


def main():
    input_sizes = {
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

    ideal_eps = {
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

    results = []
    # Analytic algorithms
    hist_pairs = generate_hist_pairs(input_sizes["NoisyHist1"])
    results.append(("NoisyHist1", input_sizes["NoisyHist1"],
                    estimate_algorithm("NoisyHist1", hist_pairs,
                                        dist_func=noisy_hist1_dist)))
    results.append(("NoisyHist2", input_sizes["NoisyHist2"],
                    estimate_algorithm("NoisyHist2", hist_pairs,
                                        dist_func=noisy_hist2_dist)))

    vec_pairs = generate_change_one_pairs(input_sizes["ReportNoisyMax1"])
    results.append(("ReportNoisyMax1", input_sizes["ReportNoisyMax1"],
                    estimate_algorithm("ReportNoisyMax1", vec_pairs,
                                       dist_func=report_noisy_max1_dist)))
    results.append(("ReportNoisyMax3", input_sizes["ReportNoisyMax3"],
                    estimate_algorithm("ReportNoisyMax3", vec_pairs,
                                       dist_func=report_noisy_max3_dist)))

    laplace_pairs = generate_change_one_pairs(input_sizes["LaplaceMechanism"])
    results.append(("LaplaceMechanism", input_sizes["LaplaceMechanism"],
                    estimate_algorithm("LaplaceMechanism", laplace_pairs,
                                       dist_func=laplace_vec_dist)))
    results.append(("LaplaceParallel", input_sizes["LaplaceParallel"],
                    estimate_algorithm(
                        "LaplaceParallel", [laplace_pairs[0]],
                        dist_func=lambda data, eps: laplace_parallel_dist(data, 0.005,
                                                                          input_sizes["LaplaceParallel"]))))

    # Algorithms computed analytically
    results.append(("ReportNoisyMax2", input_sizes["ReportNoisyMax2"],
                    estimate_algorithm("ReportNoisyMax2", vec_pairs,
                                       dist_func=report_noisy_max2_dist)))
    results.append(("ReportNoisyMax4", input_sizes["ReportNoisyMax4"],
                    estimate_algorithm("ReportNoisyMax4", vec_pairs,
                                       dist_func=report_noisy_max4_dist)))

    svt_pairs_short = generate_change_one_pairs(input_sizes["SVT1"])
    svt_pairs_long = generate_change_one_pairs(input_sizes["SVT5"])
    results.append(("SVT1", input_sizes["SVT1"],
                    estimate_algorithm("SVT1", svt_pairs_short,
                                       mechanism=SparseVectorTechnique1(eps=0.1))))
    results.append(("SVT2", input_sizes["SVT2"],
                    estimate_algorithm("SVT2", svt_pairs_short,
                                       mechanism=SparseVectorTechnique2(eps=0.1))))
    results.append(("SVT3", input_sizes["SVT3"],
                    estimate_algorithm("SVT3", svt_pairs_short,
                                       mechanism=SparseVectorTechnique3(eps=0.1))))
    results.append(("SVT4", input_sizes["SVT4"],
                    estimate_algorithm("SVT4", svt_pairs_short,
                                       mechanism=SparseVectorTechnique4(eps=0.1))))
    results.append(("SVT5", input_sizes["SVT5"],
                    estimate_algorithm("SVT5", svt_pairs_long,
                                       mechanism=SparseVectorTechnique5(eps=0.1))))
    results.append(("SVT6", input_sizes["SVT6"],
                    estimate_algorithm("SVT6", svt_pairs_long,
                                       mechanism=SparseVectorTechnique6(eps=0.1))))

    results.append(("NumericalSVT", input_sizes["NumericalSVT"],
                    estimate_algorithm("NumericalSVT",
                                       generate_change_one_pairs(input_sizes["NumericalSVT"]),
                                       mechanism=NumericalSVT(eps=0.1))))

    prefix_pairs = generate_change_one_pairs(input_sizes["PrefixSum"])
    results.append(("PrefixSum", input_sizes["PrefixSum"],
                    estimate_algorithm("PrefixSum", prefix_pairs,
                                       mechanism=PrefixSum(eps=0.1))))

    otr_pairs = generate_change_one_pairs(input_sizes["OneTimeRAPPOR"])
    results.append(("OneTimeRAPPOR", input_sizes["OneTimeRAPPOR"],
                    estimate_algorithm("OneTimeRAPPOR", otr_pairs,
                                       mechanism=OneTimeRappor())))

    rappor_pairs = generate_change_one_pairs(input_sizes["RAPPOR"])
    results.append(("RAPPOR", input_sizes["RAPPOR"],
                    estimate_algorithm("RAPPOR", rappor_pairs,
                                       mechanism=Rappor())))

    results.append(("SVT34Parallel", input_sizes["SVT34Parallel"],
                    estimate_algorithm("SVT34Parallel", svt_pairs_long,
                                       mechanism=SVT34Parallel(eps=0.1))))

    tg_pairs = [(np.array([2]), np.array([1])), (np.array([1]), np.array([0]))]
    results.append(("TruncatedGeometric", input_sizes["TruncatedGeometric"],
                    estimate_algorithm(
                        "TruncatedGeometric", tg_pairs,
                        mechanism=TruncatedGeometricMechanism(eps=0.1, n=5))))

    # Write markdown report
    with open("docs/privacy_loss_report.md", "w") as f:
        f.write("# Privacy Loss Report\n\n")
        f.write("| Algorithm | Input size | Estimated ε | Ideal ε |\n")
        f.write("|-----------|------------|-------------|---------|\n")
        for name, size, eps in results:
            ideal = ideal_eps.get(name)
            if ideal is None:
                ideal_str = "N/A"
            elif math.isinf(ideal):
                ideal_str = "∞"
            else:
                ideal_str = f"{ideal:.2f}"
            f.write(f"| {name} | {size} | {eps:.4f} | {ideal_str} |\n")

    for name, size, eps in results:
        ideal = ideal_eps.get(name)
        if ideal is not None:
            ideal_display = "∞" if math.isinf(ideal) else f"{ideal:.2f}"
            print(f"{name} (n={size}): ε ≈ {eps:.4f} (ideal {ideal_display})")
        else:
            print(f"{name} (n={size}): ε ≈ {eps:.4f}")

if __name__ == "__main__":
    main()
