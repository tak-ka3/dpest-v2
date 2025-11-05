"""Distribution-based mechanism helpers used in examples/tests."""

from __future__ import annotations

from typing import List

try:
    import mmh3  # optional dependency
except ModuleNotFoundError:  # pragma: no cover - optional
    mmh3 = None

import numpy as np

from ..core import Dist
from ..engine import AlgorithmBuilder, vector_argmax, vector_max
from ..noise import create_exponential_noise, create_laplace_noise
from ..operations import Condition, add


def noisy_hist1_dist(a: np.ndarray, eps: float) -> List[Dist]:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=1 / eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)


def noisy_hist2_dist(a: np.ndarray, eps: float) -> List[Dist]:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)


def report_noisy_max1_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=2 / eps, size=len(a))
    return vector_argmax(AlgorithmBuilder.vector_add(x_dists, noise_dists))


def report_noisy_max3_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=2 / eps, size=len(a))
    return vector_max(AlgorithmBuilder.vector_add(x_dists, noise_dists))


def report_noisy_max2_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_exponential_noise(b=2 / eps, size=len(a))
    dist = vector_argmax(AlgorithmBuilder.vector_add(x_dists, noise_dists))
    dist.normalize()
    return dist


def report_noisy_max4_dist(a: np.ndarray, eps: float) -> Dist:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_exponential_noise(b=2 / eps, size=len(a))
    dist = vector_max(AlgorithmBuilder.vector_add(x_dists, noise_dists))
    dist.normalize()
    return dist


def laplace_vec_dist(a: np.ndarray, eps: float) -> List[Dist]:
    x_dists = [Dist.deterministic(float(v)) for v in a]
    noise_dists = create_laplace_noise(b=1 / eps, size=len(a))
    return AlgorithmBuilder.vector_add(x_dists, noise_dists)


def laplace_parallel_dist(a: np.ndarray, eps_each: float, n_parallel: int) -> List[Dist]:
    x_dist = Dist.deterministic(float(a.item(0)))
    noise_list = create_laplace_noise(b=1 / eps_each, size=n_parallel)
    return [add(x_dist, n) for n in noise_list]


def one_time_rappor_dist(
    a: np.ndarray,
    eps: float,
    n_hashes: int = 4,
    filter_size: int = 20,
    f: float = 0.95,
) -> List[Dist]:
    if mmh3 is None:  # pragma: no cover - optional
        raise ModuleNotFoundError("mmh3 is required for RAPPOR distributions")

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
    if mmh3 is None:  # pragma: no cover - optional
        raise ModuleNotFoundError("mmh3 is required for RAPPOR distributions")

    perm_dists = one_time_rappor_dist(a, eps, n_hashes=n_hashes, filter_size=filter_size, f=f)
    dist_if_one = Dist.from_atoms([(1.0, q), (0.0, 1.0 - q)])
    dist_if_zero = Dist.from_atoms([(1.0, p), (0.0, 1.0 - p)])
    dists: List[Dist] = []
    for perm in perm_dists:
        final = Condition.apply(perm, dist_if_one, dist_if_zero)
        dists.append(final)
    return dists


__all__ = [
    'noisy_hist1_dist', 'noisy_hist2_dist',
    'report_noisy_max1_dist', 'report_noisy_max2_dist',
    'report_noisy_max3_dist', 'report_noisy_max4_dist',
    'laplace_vec_dist', 'laplace_parallel_dist',
    'one_time_rappor_dist', 'rappor_dist',
]
