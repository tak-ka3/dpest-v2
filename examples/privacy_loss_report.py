import os
import sys
import math
from dataclasses import dataclass
import numpy as np
import mmh3
from typing import Dict, List, Tuple, Optional

# このファイルを直接実行する際に dpest パッケージを読み込めるようにする
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dpest.core import Dist
from dpest.engine import AlgorithmBuilder, vector_argmax, vector_max
from dpest.operations import (
    add_distributions,
    compare_geq,
    Condition,
    SharedThresholdGrid,
    SharedThresholdIntegrator,
    ThresholdBranch,
)
from dpest.noise import create_laplace_noise, create_exponential_noise
from dpest.utils.input_patterns import generate_patterns

from dpest.mechanisms.sparse_vector_technique import NumericalSVT
from dpest.mechanisms.parallel import SVT34Parallel
from dpest.mechanisms.prefix_sum import PrefixSum
from dpest.mechanisms.geometric import TruncatedGeometricMechanism
from dpest.utils.privacy import (
    epsilon_from_dist,
    epsilon_from_list,
    epsilon_from_samples_matrix,
)

# ---------------------------------------------------------------------------
# 演算を用いた解析的実装
# ---------------------------------------------------------------------------

def noisy_hist1_dist(a: np.ndarray, eps: float) -> List[Dist]:
    """各要素にラプラスノイズ (b=1/eps) を加えたヒストグラムを返す。"""
    # 入力値を決定的な分布に変換
    x_dists = [Dist.deterministic(float(v)) for v in a]
    # 独立なラプラスノイズを生成
    noise_dists = create_laplace_noise(b=1 / eps, size=len(a))
    # ベクトルとして各要素にノイズを加算
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)

def noisy_hist2_dist(a: np.ndarray, eps: float) -> List[Dist]:
    """ノイズスケール ``eps`` のラプラスノイズを各要素に加える。"""
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)

def report_noisy_max1_dist(a: np.ndarray, eps: float) -> Dist:
    """ラプラスノイズを加えた後の argmax の分布を計算する。"""
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=2 / eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    return vector_argmax(z_dists)

def report_noisy_max3_dist(a: np.ndarray, eps: float) -> Dist:
    """ラプラスノイズ後の最大値そのものの分布を求める。"""
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=2 / eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    return vector_max(z_dists)

def report_noisy_max2_dist(a: np.ndarray, eps: float) -> Dist:
    """指数ノイズを用いた argmax の分布。"""
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_exponential_noise(b=2 / eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    dist = vector_argmax(z_dists)
    dist.normalize()
    return dist

def report_noisy_max4_dist(a: np.ndarray, eps: float) -> Dist:
    """指数ノイズ後の最大値の分布。"""
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_exponential_noise(b=2 / eps, size=len(a))
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    dist = vector_max(z_dists)
    dist.normalize()
    return dist

def laplace_vec_dist(a: np.ndarray, eps: float) -> List[Dist]:
    """入力ベクトルに要素ごとラプラス機構を適用する。"""
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=1 / eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)

def laplace_parallel_dist(a: np.ndarray, eps_each: float, n_parallel: int) -> List[Dist]:
    """同じ入力に独立なラプラス機構を並列に適用する。"""
    x_dist = Dist.deterministic(float(a.item(0)))
    noise_list = create_laplace_noise(b=1 / eps_each, size=n_parallel)
    return [add_distributions(x_dist, n) for n in noise_list]


def svt1_joint_dist(
    a: np.ndarray,
    eps: float,
    c: int = 2,
    t: float = 1.0,
    grid_size: int = 1000,
) -> Dist:
    """Return joint output distribution of SVT1 using primitive operations."""

    x = np.atleast_1d(a)
    if eps <= 0:
        raise ValueError("eps must be positive")

    eps1 = eps / 2.0
    eps2 = eps - eps1

    if c < 0:
        raise ValueError("c must be non-negative")

    rho_dist = create_laplace_noise(b=1 / eps1, grid_size=grid_size)
    thresh_dist = add_distributions(Dist.deterministic(t), rho_dist)

    nu_scale = 2.0 * c / eps2 if len(x) > 0 else 0.0

    threshold_grid = SharedThresholdGrid.from_dist(thresh_dist)
    integrator = SharedThresholdIntegrator(threshold_grid)
    true_value = 1.0
    false_value = 0.0
    abort_value = -1.0

    @dataclass(frozen=True)
    class SVT1State:
        true_count: int
        halted: bool

    states = integrator.initial_states(
        SVT1State(true_count=0, halted=(c <= 0))
    )

    for query_value in x:
        # Follow Algorithm 1: compare the noisy query against the shared
        # threshold and branch into TRUE, FALSE, or ABORT outcomes.

        def transition(seq, state, grid=threshold_grid, value=float(query_value)):
            status: SVT1State = state.payload
            if status.halted:
                yield ThresholdBranch(symbol=abort_value, event_prob=grid.ones(), payload=status)
                return

            diff = grid.values - value
            positive = diff >= 0.0
            p_true = np.empty_like(diff, dtype=float)
            # Laplace survival for ``ν >= diff`` with ν ~ Lap(0, b)
            if nu_scale <= 0.0:
                p_true.fill(1.0 if np.all(diff <= 0.0) else 0.0)
            else:
                p_true[positive] = 0.5 * np.exp(-diff[positive] / nu_scale)
                negative = ~positive
                p_true[negative] = 1.0 - 0.5 * np.exp(diff[negative] / nu_scale)
            p_true = np.clip(p_true, 0.0, 1.0)
            p_false = np.clip(1.0 - p_true, 0.0, 1.0)

            if np.any(p_true > 0.0):
                next_count = status.true_count + 1
                yield ThresholdBranch(
                    symbol=true_value,
                    event_prob=p_true,
                    payload=SVT1State(next_count, halted=(next_count >= c)),
                )

            if np.any(p_false > 0.0):
                yield ThresholdBranch(
                    symbol=false_value,
                    event_prob=p_false,
                    payload=status,
                )

        states = integrator.step(states, transition)

    dist = integrator.finalize(states)
    return dist


def svt2_joint_dist(a: np.ndarray, eps: float, c: int = 2, t: float = 1.0) -> Dist:
    """Return joint output distribution of SVT2 using basic operations."""

    x = np.atleast_1d(a)
    if eps <= 0:
        raise ValueError("eps must be positive")

    eps1 = eps / 2.0
    eps2 = eps - eps1

    # Initial threshold noise ρ₀ ~ Lap(c/ε₁)
    rho_init = create_laplace_noise(b=c / eps1)
    thresh_init = add_distributions(Dist.deterministic(t), rho_init)

    # Threshold noise after TRUE outputs ρ' ~ Lap(c/ε₂)
    rho_reset = create_laplace_noise(b=c / eps2)
    thresh_reset = add_distributions(Dist.deterministic(t), rho_reset)

    # Noise for each query ν_i ~ Lap(2c/ε₂)
    nu_scale = 2.0 * c / eps2
    nu_dists = create_laplace_noise(b=nu_scale, size=len(x))

    sequences: Dict[Tuple[float, ...], Tuple[float, int, Dist]] = {
        (): (1.0, 0, thresh_init)
    }

    for val, nu in zip(x, nu_dists):
        new_sequences: Dict[Tuple[float, ...], Tuple[float, int, Dist]] = {}
        for seq, (prob, k, thresh_dist) in sequences.items():
            if k >= c:
                new_seq = seq + (-1.0,)
                new_sequences[new_seq] = (
                    new_sequences.get(new_seq, (0.0, k, thresh_dist))[0] + prob,
                    k,
                    thresh_dist,
                )
                continue

            val_dist = add_distributions(Dist.deterministic(float(val)), nu)
            cmp_dist = compare_geq(val_dist, thresh_dist)
            p_true = next((w for v, w in cmp_dist.atoms if v == 1.0), 0.0)
            p_false = next((w for v, w in cmp_dist.atoms if v == 0.0), 0.0)

            if p_true > 0:
                seq_true = seq + (1.0,)
                new_sequences[seq_true] = (
                    new_sequences.get(seq_true, (0.0, k + 1, thresh_reset))[0]
                    + prob * p_true,
                    k + 1,
                    thresh_reset,
                )
            if p_false > 0:
                seq_false = seq + (0.0,)
                new_sequences[seq_false] = (
                    new_sequences.get(seq_false, (0.0, k, thresh_dist))[0]
                    + prob * p_false,
                    k,
                    thresh_dist,
                )
        sequences = new_sequences

    atoms = [(seq, prob) for seq, (prob, _, _) in sequences.items()]
    dist = Dist.from_atoms(atoms)
    dist.normalize()
    return dist


def svt3_joint_dist(
    a: np.ndarray,
    eps: float,
    c: int = 2,
    t: float = 1.0,
) -> Dist:
    """Return joint output distribution of SVT3 using basic operations.

    The joint distribution is formed by enumerating all possible sequences of
    comparisons against a shared noisy threshold.  Each output is categorized
    as TRUE (1), FALSE (0), or ABORTED (-1) after ``c`` TRUE answers.
    """

    x = np.atleast_1d(a)
    eps1 = eps / 2.0
    eps2 = eps - eps1

    # Shared noisy threshold ρ ~ Lap(1/ε₁)
    rho_dist = create_laplace_noise(b=1 / eps1)
    thresh_dist = add_distributions(Dist.deterministic(t), rho_dist)

    # Noise for each query ν_i ~ Lap(c/ε₂)
    nu_dists = create_laplace_noise(b=c / eps2, size=len(x))

    sequences: Dict[Tuple[float, ...], Tuple[float, int]] = {(): (1.0, 0)}

    for val, nu in zip(x, nu_dists):
        val_dist = add_distributions(Dist.deterministic(float(val)), nu)
        cmp_dist = compare_geq(val_dist, thresh_dist)
        p_true = next((w for v, w in cmp_dist.atoms if v == 1.0), 0.0)
        p_false = next((w for v, w in cmp_dist.atoms if v == 0.0), 0.0)

        new_sequences: Dict[Tuple[float, ...], Tuple[float, int]] = {}
        for seq, (prob, k) in sequences.items():
            if k >= c:
                new_seq = seq + (-1.0,)
                new_sequences[new_seq] = (
                    new_sequences.get(new_seq, (0.0, k))[0] + prob,
                    k,
                )
                continue
            if p_true > 0:
                seq_true = seq + (1.0,)
                new_sequences[seq_true] = (
                    new_sequences.get(seq_true, (0.0, k + 1))[0] + prob * p_true,
                    k + 1,
                )
            if p_false > 0:
                seq_false = seq + (0.0,)
                new_sequences[seq_false] = (
                    new_sequences.get(seq_false, (0.0, k))[0] + prob * p_false,
                    k,
                )
        sequences = new_sequences

    atoms = [(seq, prob) for seq, (prob, _) in sequences.items()]
    dist = Dist.from_atoms(atoms)
    dist.normalize()
    return dist


def svt4_joint_dist(
    a: np.ndarray,
    eps: float,
    c: int = 2,
    t: float = 1.0,
) -> Dist:
    """Return joint output distribution of SVT4 using basic operations."""

    x = np.atleast_1d(a)
    eps1 = eps / 4.0
    eps2 = eps - eps1

    # Shared noisy threshold ρ ~ Lap(1/ε₁)
    rho_dist = create_laplace_noise(b=1 / eps1)
    thresh_dist = add_distributions(Dist.deterministic(t), rho_dist)

    # Noise for each query ν_i ~ Lap(1/ε₂)
    nu_dists = create_laplace_noise(b=1 / eps2, size=len(x))

    sequences: Dict[Tuple[float, ...], Tuple[float, int]] = {(): (1.0, 0)}

    for val, nu in zip(x, nu_dists):
        val_dist = add_distributions(Dist.deterministic(float(val)), nu)
        cmp_dist = compare_geq(val_dist, thresh_dist)
        p_true = next((w for v, w in cmp_dist.atoms if v == 1.0), 0.0)
        p_false = next((w for v, w in cmp_dist.atoms if v == 0.0), 0.0)

        new_sequences: Dict[Tuple[float, ...], Tuple[float, int]] = {}
        for seq, (prob, k) in sequences.items():
            if k >= c:
                new_seq = seq + (-1.0,)
                new_sequences[new_seq] = (
                    new_sequences.get(new_seq, (0.0, k))[0] + prob,
                    k,
                )
                continue
            if p_true > 0:
                seq_true = seq + (1.0,)
                new_sequences[seq_true] = (
                    new_sequences.get(seq_true, (0.0, k + 1))[0] + prob * p_true,
                    k + 1,
                )
            if p_false > 0:
                seq_false = seq + (0.0,)
                new_sequences[seq_false] = (
                    new_sequences.get(seq_false, (0.0, k))[0] + prob * p_false,
                    k,
                )
        sequences = new_sequences

    atoms = [(seq, prob) for seq, (prob, _) in sequences.items()]
    dist = Dist.from_atoms(atoms)
    dist.normalize()
    return dist


def svt6_joint_dist(
    a: np.ndarray,
    eps: float,
    t: float = 1.0,
    grid_size: int = 1000,
) -> Dist:
    """Return joint output distribution of SVT6."""
    # しきい値と各クエリに独立なノイズを加え、全比較結果の列挙で分布を構築
    x = np.atleast_1d(a)
    eps1 = eps / 2.0
    eps2 = eps - eps1

    rho_dist = create_laplace_noise(b=1 / eps1, grid_size=grid_size)
    noise_dists = create_laplace_noise(b=1 / eps2, size=len(x), grid_size=grid_size)

    cdf_funcs = []
    for val, noise in zip(x, noise_dists):
        val_dist = add_distributions(Dist.deterministic(float(val)), noise)
        y = val_dist.density["x"]
        f = val_dist.density["f"]
        dx = val_dist.density["dx"]
        cdf = np.cumsum(f) * dx

        def F(y_val, grid=y, cdf_vals=cdf):
            return np.interp(y_val, grid, cdf_vals, left=0.0, right=1.0)

        cdf_funcs.append(F)

    rho_x = rho_dist.density["x"]
    rho_f = rho_dist.density["f"]
    rho_dx = rho_dist.density["dx"]

    sequence_probs: Dict[Tuple[float, ...], float] = {}
    for r, weight in zip(rho_x, rho_f):
        thresh = t + r
        p_list = [1.0 - F(thresh) for F in cdf_funcs]
        seqs: Dict[Tuple[float, ...], float] = {(): 1.0}
        for p_true in p_list:
            p_false = 1.0 - p_true
            new_seqs: Dict[Tuple[float, ...], float] = {}
            for seq, prob in seqs.items():
                seq_true = seq + (1.0,)
                seq_false = seq + (0.0,)
                new_seqs[seq_true] = new_seqs.get(seq_true, 0.0) + prob * p_true
                new_seqs[seq_false] = new_seqs.get(seq_false, 0.0) + prob * p_false
            seqs = new_seqs
        final_weight = weight * rho_dx
        for seq, prob in seqs.items():
            sequence_probs[seq] = sequence_probs.get(seq, 0.0) + prob * final_weight

    atoms = [(seq, p) for seq, p in sequence_probs.items()]
    dist = Dist.from_atoms(atoms)
    dist.normalize()
    return dist


def svt5_joint_dist(
    a: np.ndarray,
    eps: float,
    t: float = 1.0,
    grid_size: int = 1000,
) -> Dist:
    """Return joint output distribution of SVT5."""
    # しきい値にノイズを加え、各クエリとの比較結果を列挙する
    x = np.atleast_1d(a)
    eps1 = eps / 2.0
    rho_dist = create_laplace_noise(b=1 / eps1, grid_size=grid_size)

    rho_x = rho_dist.density["x"]
    rho_f = rho_dist.density["f"]
    rho_dx = rho_dist.density["dx"]

    sequence_probs: Dict[Tuple[float, ...], float] = {}
    for r, weight in zip(rho_x, rho_f):
        thresh = t + r
        seq = tuple(1.0 if val >= thresh else 0.0 for val in x)
        prob = weight * rho_dx
        sequence_probs[seq] = sequence_probs.get(seq, 0.0) + prob

    atoms = [(seq, p) for seq, p in sequence_probs.items()]
    dist = Dist.from_atoms(atoms)
    dist.normalize()
    return dist


def one_time_rappor_dist(
    a: np.ndarray,
    eps: float,
    n_hashes: int = 4,
    filter_size: int = 20,
    f: float = 0.95,
) -> List[Dist]:
    """Distribution of One-time RAPPOR using analytic operations."""
    # ハッシュにより値をビットベクトルへエンコード
    val = int(a.item(0))
    filter_bits = np.zeros(filter_size, dtype=int)
    for i in range(n_hashes):
        idx = mmh3.hash(str(val), seed=i) % filter_size
        filter_bits[idx] = 1

    # f の確率でランダマイズ、1/0 それぞれの分布を定義
    cond_randomize = Dist.from_atoms([(1.0, f), (0.0, 1.0 - f)])
    cond_flip = Dist.from_atoms([(1.0, 0.5), (0.0, 0.5)])
    bit_one = Dist.deterministic(1.0)
    bit_zero = Dist.deterministic(0.0)
    random_bit = Condition.apply(cond_flip, bit_one, bit_zero)

    # 各ビットにランダマイズを適用
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
    # まず一回ランダマイズ版を計算
    perm_dists = one_time_rappor_dist(
        a, eps, n_hashes=n_hashes, filter_size=filter_size, f=f
    )
    # 本番のランダマイズを適用 (q:1, p:0 に flip)
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
                                       joint_dist_func=svt2_joint_dist)))
    results.append(("SVT3", input_sizes["SVT3"],
                    estimate_algorithm("SVT3", svt_pairs_short,
                                       joint_dist_func=svt3_joint_dist)))
    results.append(("SVT4", input_sizes["SVT4"],
                    estimate_algorithm("SVT4", svt_pairs_short,
                                       joint_dist_func=svt4_joint_dist)))
    results.append(("SVT5", input_sizes["SVT5"],
                    estimate_algorithm("SVT5", svt_pairs_long,
                                       joint_dist_func=svt5_joint_dist)))
    results.append(("SVT6", input_sizes["SVT6"],
                    estimate_algorithm("SVT6", svt_pairs_long,
                                       joint_dist_func=svt6_joint_dist)))

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
