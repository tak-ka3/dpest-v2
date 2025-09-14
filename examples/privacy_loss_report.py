import os
import sys
import math
import numpy as np
import mmh3
from typing import Dict, List, Tuple

# Allow importing the dpest package when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dpest.core import Dist
from dpest.engine import AlgorithmBuilder, vector_argmax, vector_max
from dpest.operations import add_distributions, compare_geq, Condition
from dpest.noise import create_laplace_noise, create_exponential_noise
from dpest.utils.input_patterns import generate_patterns

from dpest.mechanisms.sparse_vector_technique import (
    SparseVectorTechnique2, SparseVectorTechnique3,
    SparseVectorTechnique4, SparseVectorTechnique6, NumericalSVT,
)
from dpest.mechanisms.parallel import SVT34Parallel
from dpest.mechanisms.prefix_sum import PrefixSum
from dpest.mechanisms.geometric import TruncatedGeometricMechanism
from dpest.utils.privacy import (
    epsilon_from_dist,
    epsilon_from_list,
    epsilon_from_samples_matrix,
)

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


def svt1_dist(a: np.ndarray, eps: float, c: int = 2, t: float = 1.0) -> List[Dist]:
    """解析的手法でSVT1の出力分布を計算する関数"""
    # 入力配列を1次元化する
    x = np.atleast_1d(a)
    # 全体のプライバシー予算εをしきい値とクエリ用に分割
    eps1 = eps / 2.0  # しきい値計算に使うε1
    eps2 = eps - eps1  # クエリ評価に使うε2
    # しきい値用ノイズρ ~ Laplace(1/ε1)
    rho_dist = create_laplace_noise(b=1 / eps1)
    # しきい値tにρを加えた分布
    thresh_dist = add_distributions(rho_dist, Dist.deterministic(t))
    # 各クエリに加えるノイズ ~ Laplace(2c/ε2)
    noise_dists = create_laplace_noise(b=2 * c / eps2, size=len(x))

    # trueの出力回数の分布（k回trueを出した確率）を保持
    count_probs = [1.0] + [0.0] * c
    results: List[Dist] = []
    # 各クエリを順に評価
    for val, noise in zip(x, noise_dists):
        # クエリ値にノイズを加えた分布
        val_dist = add_distributions(Dist.deterministic(float(val)), noise)
        # しきい値以上かどうかの比較結果の分布
        cmp_dist = compare_geq(val_dist, thresh_dist)
        # 比較が真(1)になる確率
        p_true = next((w for v, w in cmp_dist.atoms if v == 1.0), 0.0)
        # 比較が偽(0)になる確率
        p_false = next((w for v, w in cmp_dist.atoms if v == 0.0), 0.0)
        # 既にc回trueを出力してアボートしている確率
        p_abort = count_probs[c]
        # まだアボートしていない状態で1を出力する確率
        p1 = sum(count_probs[:c]) * p_true
        # まだアボートしていない状態で0を出力する確率
        p0 = sum(count_probs[:c]) * p_false
        atoms = [(1.0, p1), (0.0, p0)]
        if p_abort > 0:
            # アボート(-1)を明示的に出力する確率
            atoms.append((-1.0, p_abort))
        # 出力の確率分布を構成
        res_dist = Dist.from_atoms(atoms)
        res_dist.normalize()
        results.append(res_dist)

        # trueの出力回数の分布を更新
        new_counts = [0.0] * (c + 1)
        for k in range(c):
            prob_k = count_probs[k]  # trueをk回出した状態の確率
            new_counts[k] += prob_k * p_false  # falseなら回数は変化なし
            new_counts[min(c, k + 1)] += prob_k * p_true  # trueなら回数+1（上限c）
        # すでにアボートしている確率を保持
        new_counts[c] += count_probs[c]
        count_probs = new_counts

    return results


def svt1_joint_dist(a: np.ndarray, eps: float, c: int = 2, t: float = 1.0) -> Dist:
    """Return joint output distribution of SVT1.

    This enumerates the probability of every output vector, inserting ``-1``
    after the mechanism aborts.  Treating the whole vector as one random
    variable captures the dependencies among coordinates.
    """
    x = np.atleast_1d(a)
    eps1 = eps / 2.0
    eps2 = eps - eps1
    rho_dist = create_laplace_noise(b=1 / eps1)
    thresh_dist = add_distributions(rho_dist, Dist.deterministic(t))
    noise_dists = create_laplace_noise(b=2 * c / eps2, size=len(x))

    p_list: List[Tuple[float, float]] = []
    for val, noise in zip(x, noise_dists):
        val_dist = add_distributions(Dist.deterministic(float(val)), noise)
        cmp_dist = compare_geq(val_dist, thresh_dist)
        p_true = next((w for v, w in cmp_dist.atoms if v == 1.0), 0.0)
        p_false = next((w for v, w in cmp_dist.atoms if v == 0.0), 0.0)
        p_list.append((p_true, p_false))

    sequences: Dict[Tuple[float, ...], float] = {(): 1.0}
    for p_true, p_false in p_list:
        new_seq: Dict[Tuple[float, ...], float] = {}
        for seq, prob in sequences.items():
            if seq and seq[-1] == -1.0:
                new_seq[seq + (-1.0,)] = new_seq.get(seq + (-1.0,), 0.0) + prob
                continue
            k = seq.count(1.0)
            if k >= c:
                new_seq[seq + (-1.0,)] = new_seq.get(seq + (-1.0,), 0.0) + prob
                continue
            seq_true = seq + (1.0,)
            seq_false = seq + (0.0,)
            new_seq[seq_true] = new_seq.get(seq_true, 0.0) + prob * p_true
            new_seq[seq_false] = new_seq.get(seq_false, 0.0) + prob * p_false
        sequences = new_seq

    atoms = [(seq, p) for seq, p in sequences.items()]
    dist = Dist.from_atoms(atoms)
    dist.normalize()
    return dist


def svt5_dist(a: np.ndarray, eps: float, t: float = 1.0) -> List[Dist]:
    """Distribution of Sparse Vector Technique 5 using analytic operations."""
    x = np.atleast_1d(a)
    eps1 = eps / 2.0
    rho_dist = create_laplace_noise(b=1 / eps1)
    thresh_dist = add_distributions(rho_dist, Dist.deterministic(t))
    results: List[Dist] = []
    for val in x:
        val_dist = Dist.deterministic(float(val))
        res_dist = compare_geq(val_dist, thresh_dist)
        results.append(res_dist)
    return results


def one_time_rappor_dist(
    a: np.ndarray,
    eps: float,
    n_hashes: int = 4,
    filter_size: int = 20,
    f: float = 0.95,
) -> List[Dist]:
    """Distribution of One-time RAPPOR using analytic operations."""
    val = int(a.item(0))
    filter_bits = np.zeros(filter_size, dtype=int)
    for i in range(n_hashes):
        idx = mmh3.hash(str(val), seed=i) % filter_size
        filter_bits[idx] = 1

    cond_randomize = Dist.from_atoms([(1.0, f), (0.0, 1.0 - f)])
    cond_flip = Dist.from_atoms([(1.0, 0.5), (0.0, 0.5)])
    bit_one = Dist.deterministic(1.0)
    bit_zero = Dist.deterministic(0.0)
    random_bit = Condition.apply(cond_flip, bit_one, bit_zero)

    dists: List[Dist] = []
    for bit in filter_bits:
        base = Dist.deterministic(float(bit))
        perm = Condition.apply(cond_randomize, random_bit, base)
        dists.append(perm)
    return dists


def rappor_dist(
    a: np.ndarray,
    eps: float,
    n_hashes: int = 4,
    filter_size: int = 20,
    f: float = 0.75,
    p: float = 0.45,
    q: float = 0.55,
) -> List[Dist]:
    """Distribution of full RAPPOR using analytic operations."""
    perm_dists = one_time_rappor_dist(
        a, eps, n_hashes=n_hashes, filter_size=filter_size, f=f
    )
    dist_if_one = Dist.from_atoms([(1.0, q), (0.0, 1.0 - q)])
    dist_if_zero = Dist.from_atoms([(1.0, p), (0.0, 1.0 - p)])
    dists: List[Dist] = []
    for perm in perm_dists:
        final = Condition.apply(perm, dist_if_one, dist_if_zero)
        dists.append(final)
    return dists

# ---------------------------------------------------------------------------
# Estimation driver
# ---------------------------------------------------------------------------

def estimate_algorithm(
    name: str,
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    *,
    dist_func=None,
    joint_dist_func=None,
    mechanism=None,
    eps: float = 0.1,
    n_samples: int = 100000,
    extra=None,
) -> float:
    """Estimate privacy loss for an algorithm.

    If ``joint_dist_func`` is provided, it is assumed to return the joint
    output distribution and dependencies between coordinates are respected.
    Otherwise lists of marginals are summed using ``epsilon_from_list``.
    """
    eps_max = 0.0
    for D, Dp in pairs:
        if joint_dist_func is not None:
            if extra is None:
                P = joint_dist_func(D, eps)
                Q = joint_dist_func(Dp, eps)
            else:
                P = joint_dist_func(D, eps, *extra)
                Q = joint_dist_func(Dp, eps, *extra)
            eps_val = epsilon_from_dist(P, Q)
        elif dist_func is not None:
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
        eps_max = max(eps_max, eps_val)
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
                                       joint_dist_func=svt1_joint_dist)))
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
                                       dist_func=svt5_dist)))
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
                                       dist_func=one_time_rappor_dist)))

    rappor_pairs = generate_change_one_pairs(input_sizes["RAPPOR"])
    results.append(("RAPPOR", input_sizes["RAPPOR"],
                    estimate_algorithm("RAPPOR", rappor_pairs,
                                       dist_func=rappor_dist)))

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
