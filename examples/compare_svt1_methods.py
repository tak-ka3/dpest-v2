"""Generate runtime and privacy loss comparisons for SVT1 implementations.

This script evaluates the analytic joint distribution constructor
``svt1_joint_dist`` and the sampling-based ``SparseVectorTechnique1``
implementation.  It sweeps several parameters and visualises how runtime and
privacy loss change for both approaches.

The following experiments are performed:

* Vary the number of queries ``m`` with a fixed TRUE budget ``c``.
* Vary the TRUE budget ``c`` while keeping the number of queries fixed.
* Inspect the effect of the Laplace discretisation grid size used by the
  analytic method.
* Inspect the impact of the sampling count used in the Monte Carlo
  approximation.

The resulting figures are written to ``examples/figures`` relative to the
repository root.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib
import numpy as np

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from examples.privacy_loss_report import svt1_joint_dist
from dpest.core import Dist
from dpest.mechanisms.sparse_vector_technique import SparseVectorTechnique1


def dist_to_prob_map(dist: Dist) -> Dict[Tuple[float, ...], float]:
    """Convert a distribution of tuples into a probability map."""

    probs: Dict[Tuple[float, ...], float] = {}
    for value, weight in dist.atoms:
        if isinstance(value, np.ndarray):
            key = tuple(float(x) for x in value.tolist())
        elif isinstance(value, (list, tuple)):
            key = tuple(float(x) for x in value)
        else:
            key = (float(value),)
        probs[key] = probs.get(key, 0.0) + float(weight)

    total = sum(probs.values())
    if total > 0:
        probs = {key: weight / total for key, weight in probs.items()}
    return probs


def sample_to_prob_map(samples: np.ndarray) -> Dict[Tuple[float, ...], float]:
    """Turn SVT output samples into an empirical distribution."""

    counts: Dict[Tuple[float, ...], int] = {}
    for row in samples:
        key = tuple(float(v) for v in row.tolist())
        counts[key] = counts.get(key, 0) + 1

    total = samples.shape[0]
    return {key: count / total for key, count in counts.items()}


def privacy_loss_from_maps(
    p: Dict[Tuple[float, ...], float],
    q: Dict[Tuple[float, ...], float],
    *,
    min_prob: float = 1e-12,
) -> float:
    """Compute the privacy loss ``ε`` between two discrete probability maps."""

    # ``min_prob`` provides a small smoothing factor so that missing outcomes in
    # empirical approximations (e.g., Monte Carlo samples) do not yield infinite
    # privacy loss even though the analytic reference assigns them positive
    # probability.

    support = set(p) | set(q)
    ratios = []
    for key in support:
        p_prob = max(p.get(key, 0.0), min_prob)
        q_prob = max(q.get(key, 0.0), min_prob)
        ratios.append(p_prob / q_prob)
        ratios.append(q_prob / p_prob)

    if not ratios:
        return 0.0

    return float(math.log(max(ratios)))


def _format_cell(value: object) -> str:
    """Format numeric values for tabular display."""

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


def _print_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Pretty-print a table to stdout."""

    if title:
        print(title)

    col_widths = []
    num_columns = len(headers)
    for idx in range(num_columns):
        width = len(headers[idx])
        if rows:
            width = max(width, *(len(_format_cell(row[idx])) for row in rows))
        col_widths.append(width)

    header_line = " | ".join(
        headers[idx].ljust(col_widths[idx]) for idx in range(num_columns)
    )
    separator = "-+-".join("-" * col_width for col_width in col_widths)

    print(header_line)
    print(separator)
    for row in rows:
        print(
            " | ".join(
                _format_cell(row[idx]).ljust(col_widths[idx]) for idx in range(num_columns)
            )
        )
    print()


def write_table(
    *,
    title: str,
    headers: Sequence[str],
    columns: Sequence[Sequence[object]],
    output_path: Path,
) -> None:
    """Write tabular results to CSV and print them."""

    rows = list(zip(*columns))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)

    _print_table(title, headers, rows)
    print(f"Saved table to {output_path}")
    print()


def run_analytic(
    a: np.ndarray,
    eps: float,
    c: int,
    t: float,
    grid_size: int,
) -> Tuple[Dist, float]:
    """Run the analytic joint distribution constructor and measure runtime."""

    start = time.perf_counter()
    dist = svt1_joint_dist(a, eps=eps, c=c, t=t, grid_size=grid_size)
    elapsed = time.perf_counter() - start
    return dist, elapsed


def run_sampling(
    a: np.ndarray,
    eps: float,
    c: int,
    t: float,
    n_samples: int,
) -> Tuple[Dict[Tuple[float, ...], float], float]:
    """Run the Monte Carlo SVT implementation and return an empirical pmf."""

    mechanism = SparseVectorTechnique1(eps=eps, c=c, t=t)
    start = time.perf_counter()
    samples = mechanism.m(a, n_samples=n_samples)
    elapsed = time.perf_counter() - start
    return sample_to_prob_map(samples), elapsed


def experiment_vary_m(
    m_values: Sequence[int],
    *,
    eps: float,
    c: int,
    t: float,
    grid_size: int,
    grid_size_ref: int,
    n_samples: int,
) -> Dict[str, Sequence[float]]:
    """Evaluate runtime and privacy loss while varying the number of queries."""

    analytic_times = []
    analytic_privacy_losses = []
    sampling_times = []
    sampling_privacy_losses = []

    for m in m_values:
        a = np.linspace(-0.5, 0.5, m)
        dist_ref, _ = run_analytic(a, eps=eps, c=c, t=t, grid_size=grid_size_ref)
        ref_map = dist_to_prob_map(dist_ref)

        dist, analytic_time = run_analytic(a, eps=eps, c=c, t=t, grid_size=grid_size)
        analytic_map = dist_to_prob_map(dist)
        analytic_privacy = privacy_loss_from_maps(ref_map, analytic_map)

        sample_map, sample_time = run_sampling(a, eps=eps, c=c, t=t, n_samples=n_samples)
        sampling_privacy = privacy_loss_from_maps(ref_map, sample_map)

        analytic_times.append(analytic_time)
        analytic_privacy_losses.append(analytic_privacy)
        sampling_times.append(sample_time)
        sampling_privacy_losses.append(sampling_privacy)

    return {
        "m_values": list(m_values),
        "analytic_times": analytic_times,
        "analytic_privacy_losses": analytic_privacy_losses,
        "sampling_times": sampling_times,
        "sampling_privacy_losses": sampling_privacy_losses,
    }


def experiment_vary_c(
    c_values: Sequence[int],
    *,
    m: int,
    eps: float,
    t: float,
    grid_size: int,
    grid_size_ref: int,
    n_samples: int,
) -> Dict[str, Sequence[float]]:
    """Evaluate runtime and privacy loss while varying the TRUE budget ``c``."""

    analytic_times = []
    analytic_privacy_losses = []
    sampling_times = []
    sampling_privacy_losses = []

    a = np.linspace(-0.5, 0.5, m)

    for c in c_values:
        dist_ref, _ = run_analytic(a, eps=eps, c=c, t=t, grid_size=grid_size_ref)
        ref_map = dist_to_prob_map(dist_ref)

        dist, analytic_time = run_analytic(a, eps=eps, c=c, t=t, grid_size=grid_size)
        analytic_map = dist_to_prob_map(dist)
        analytic_privacy = privacy_loss_from_maps(ref_map, analytic_map)

        sample_map, sample_time = run_sampling(a, eps=eps, c=c, t=t, n_samples=n_samples)
        sampling_privacy = privacy_loss_from_maps(ref_map, sample_map)

        analytic_times.append(analytic_time)
        analytic_privacy_losses.append(analytic_privacy)
        sampling_times.append(sample_time)
        sampling_privacy_losses.append(sampling_privacy)

    return {
        "c_values": list(c_values),
        "analytic_times": analytic_times,
        "analytic_privacy_losses": analytic_privacy_losses,
        "sampling_times": sampling_times,
        "sampling_privacy_losses": sampling_privacy_losses,
    }


def experiment_grid_size(
    grid_sizes: Sequence[int],
    *,
    m: int,
    eps: float,
    c: int,
    t: float,
) -> Dict[str, Sequence[float]]:
    """Inspect runtime and privacy loss for different analytic grid resolutions."""

    a = np.linspace(-0.5, 0.5, m)
    reference_size = max(grid_sizes)
    dist_ref, _ = run_analytic(a, eps=eps, c=c, t=t, grid_size=reference_size)
    ref_map = dist_to_prob_map(dist_ref)

    times = []
    privacy_losses = []
    for grid_size in grid_sizes:
        dist, elapsed = run_analytic(a, eps=eps, c=c, t=t, grid_size=grid_size)
        analytic_map = dist_to_prob_map(dist)
        privacy_loss = privacy_loss_from_maps(ref_map, analytic_map)
        times.append(elapsed)
        privacy_losses.append(privacy_loss)

    return {
        "grid_sizes": list(grid_sizes),
        "times": times,
        "privacy_losses": privacy_losses,
        "reference_size": reference_size,
    }


def experiment_sampling_counts(
    sample_counts: Sequence[int],
    *,
    m: int,
    eps: float,
    c: int,
    t: float,
    grid_size: int,
) -> Dict[str, Sequence[float]]:
    """Inspect runtime and privacy loss for different sampling counts."""

    a = np.linspace(-0.5, 0.5, m)
    dist_ref, analytic_time = run_analytic(a, eps=eps, c=c, t=t, grid_size=grid_size)
    ref_map = dist_to_prob_map(dist_ref)

    times = []
    privacy_losses = []
    for n_samples in sample_counts:
        sample_map, elapsed = run_sampling(a, eps=eps, c=c, t=t, n_samples=n_samples)
        privacy_loss = privacy_loss_from_maps(ref_map, sample_map)
        times.append(elapsed)
        privacy_losses.append(privacy_loss)

    return {
        "sample_counts": list(sample_counts),
        "times": times,
        "privacy_losses": privacy_losses,
        "analytic_time": analytic_time,
    }


def plot_runtime_accuracy(
    x_values: Sequence[int],
    analytic_times: Sequence[float],
    sampling_times: Sequence[float],
    analytic_privacy_losses: Sequence[float],
    sampling_privacy_losses: Sequence[float],
    *,
    xlabel: str,
    title: str,
    output_path: Path,
):
    """Render a two-row plot showing runtime and privacy loss."""

    fig, (ax_time, ax_privacy) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_time.plot(x_values, analytic_times, marker="o", label="Analytic")
    ax_time.plot(x_values, sampling_times, marker="s", label="Sampling")
    ax_time.set_ylabel("Runtime [s]")
    ax_time.set_title(title)
    ax_time.grid(True, which="both", axis="both", linestyle=":")
    ax_time.legend()

    ax_privacy.plot(
        x_values, analytic_privacy_losses, marker="o", label="Analytic vs ref"
    )
    ax_privacy.plot(
        x_values, sampling_privacy_losses, marker="s", label="Sampling vs ref"
    )
    ax_privacy.set_xlabel(xlabel)
    ax_privacy.set_ylabel(r"Privacy loss $\varepsilon$")
    ax_privacy.grid(True, which="both", axis="both", linestyle=":")
    ax_privacy.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_grid_size(
    grid_sizes: Sequence[int],
    times: Sequence[float],
    privacy_losses: Sequence[float],
    *,
    output_path: Path,
):
    """Plot runtime and privacy loss as functions of the analytic grid size."""

    fig, (ax_time, ax_privacy) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_time.plot(grid_sizes, times, marker="o")
    ax_time.set_ylabel("Runtime [s]")
    ax_time.set_title("Analytic method: effect of Laplace grid size")
    ax_time.grid(True, linestyle=":")

    ax_privacy.plot(grid_sizes, privacy_losses, marker="o")
    ax_privacy.set_xlabel("Laplace grid size")
    ax_privacy.set_ylabel(
        r"Privacy loss $\varepsilon$ vs grid={}".format(max(grid_sizes))
    )
    ax_privacy.grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_sampling_counts(
    sample_counts: Sequence[int],
    times: Sequence[float],
    privacy_losses: Sequence[float],
    *,
    analytic_time: float,
    output_path: Path,
):
    """Plot runtime and privacy loss for different sampling counts."""

    fig, (ax_time, ax_privacy) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_time.plot(sample_counts, times, marker="s", label="Sampling")
    ax_time.axhline(analytic_time, color="tab:orange", linestyle="--", label="Analytic (grid)")
    ax_time.set_xscale("log")
    ax_time.set_ylabel("Runtime [s]")
    ax_time.set_title("Sampling method: effect of sample count")
    ax_time.grid(True, which="both", linestyle=":")
    ax_time.legend()

    ax_privacy.plot(
        sample_counts, privacy_losses, marker="s", label="Sampling vs analytic"
    )
    ax_privacy.set_xscale("log")
    ax_privacy.set_xlabel("Number of Monte Carlo samples")
    ax_privacy.set_ylabel(r"Privacy loss $\varepsilon$")
    ax_privacy.grid(True, which="both", linestyle=":")
    ax_privacy.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    np.random.seed(0)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    eps = 1.0
    t = 0.0
    grid_size_default = 1000
    grid_size_reference = 4000
    n_samples_default = 30000

    m_values = [10, 20, 30, 40, 50]
    results_m = experiment_vary_m(
        m_values,
        eps=eps,
        c=2,
        t=t,
        grid_size=grid_size_default,
        grid_size_ref=grid_size_reference,
        n_samples=n_samples_default,
    )
    write_table(
        title="SVT1 runtime and privacy loss vs. number of queries",
        headers=[
            "Number of queries",
            "Analytic runtime [s]",
            "Sampling runtime [s]",
            "Analytic privacy loss ε",
            "Sampling privacy loss ε",
        ],
        columns=[
            results_m["m_values"],
            results_m["analytic_times"],
            results_m["sampling_times"],
            results_m["analytic_privacy_losses"],
            results_m["sampling_privacy_losses"],
        ],
        output_path=tables_dir / "svt1_compare_vs_m.csv",
    )
    plot_runtime_accuracy(
        results_m["m_values"],
        results_m["analytic_times"],
        results_m["sampling_times"],
        results_m["analytic_privacy_losses"],
        results_m["sampling_privacy_losses"],
        xlabel="Number of queries",
        title="SVT1 runtime and privacy loss vs. number of queries",
        output_path=output_dir / "svt1_compare_vs_m.png",
    )

    c_values = [1, 2, 3, 4]
    results_c = experiment_vary_c(
        c_values,
        m=15,
        eps=eps,
        t=t,
        grid_size=grid_size_default,
        grid_size_ref=grid_size_reference,
        n_samples=n_samples_default,
    )
    write_table(
        title="SVT1 runtime and privacy loss vs. TRUE budget",
        headers=[
            "TRUE budget c",
            "Analytic runtime [s]",
            "Sampling runtime [s]",
            "Analytic privacy loss ε",
            "Sampling privacy loss ε",
        ],
        columns=[
            results_c["c_values"],
            results_c["analytic_times"],
            results_c["sampling_times"],
            results_c["analytic_privacy_losses"],
            results_c["sampling_privacy_losses"],
        ],
        output_path=tables_dir / "svt1_compare_vs_c.csv",
    )
    plot_runtime_accuracy(
        results_c["c_values"],
        results_c["analytic_times"],
        results_c["sampling_times"],
        results_c["analytic_privacy_losses"],
        results_c["sampling_privacy_losses"],
        xlabel="TRUE budget c",
        title="SVT1 runtime and privacy loss vs. TRUE budget",
        output_path=output_dir / "svt1_compare_vs_c.png",
    )

    grid_sizes = [200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000]
    results_grid = experiment_grid_size(
        grid_sizes,
        m=20,
        eps=eps,
        c=2,
        t=t,
    )
    write_table(
        title="Analytic method: effect of Laplace grid size",
        headers=[
            "Laplace grid size",
            "Runtime [s]",
            f"Privacy loss ε vs grid={results_grid['reference_size']}",
        ],
        columns=[
            results_grid["grid_sizes"],
            results_grid["times"],
            results_grid["privacy_losses"],
        ],
        output_path=tables_dir / "svt1_analytic_grid_size.csv",
    )
    plot_grid_size(
        results_grid["grid_sizes"],
        results_grid["times"],
        results_grid["privacy_losses"],
        output_path=output_dir / "svt1_analytic_grid_size.png",
    )

    sample_counts = [1000, 3000, 10000, 30000, 100000]
    results_sampling = experiment_sampling_counts(
        sample_counts,
        m=20,
        eps=eps,
        c=2,
        t=t,
        grid_size=grid_size_reference,
    )
    write_table(
        title="Sampling method: effect of sample count",
        headers=[
            "Number of Monte Carlo samples",
            "Runtime [s]",
            "Privacy loss ε",
        ],
        columns=[
            results_sampling["sample_counts"],
            results_sampling["times"],
            results_sampling["privacy_losses"],
        ],
        output_path=tables_dir / "svt1_sampling_counts.csv",
    )
    print(
        "Analytic reference runtime for sampling comparison:",
        _format_cell(results_sampling["analytic_time"]),
    )
    print()
    plot_sampling_counts(
        results_sampling["sample_counts"],
        results_sampling["times"],
        results_sampling["privacy_losses"],
        analytic_time=results_sampling["analytic_time"],
        output_path=output_dir / "svt1_sampling_counts.png",
    )

    print("Figures written to", output_dir)
    print("Tables written to", tables_dir)


if __name__ == "__main__":
    main()

