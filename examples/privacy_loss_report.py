import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# このファイルを直接実行する際に dpest パッケージを読み込めるようにする
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dpest.analysis import estimate_algorithm, generate_change_one_pairs, generate_hist_pairs
from dpest.algorithms import (
    laplace_parallel,
    laplace_vec,
    noisy_hist1,
    noisy_hist2,
    noisy_max_sum,
    numerical_svt,
    one_time_rappor,
    rappor,
    report_noisy_max1,
    report_noisy_max2,
    report_noisy_max3,
    report_noisy_max4,
    svt1,
    svt2,
    svt3,
    svt4,
    svt5,
    svt6,
)
from dpest.mechanisms.geometric import TruncatedGeometricMechanism
from dpest.mechanisms.parallel import SVT34Parallel
from dpest.mechanisms.prefix_sum import PrefixSum


def _resolve_dist_func(obj):
    dist_func = getattr(obj, "_dist_func", None)
    if dist_func is None:
        return obj
    return dist_func


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "privacy_loss_report_config.json",
)


def load_config(path: Optional[str]) -> Dict[str, int]:
    """設定ファイルからサンプル数とビン数を読み込む."""

    config = {"n_samples": 100_000, "hist_bins": 100}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse config file '{path}': {exc}") from exc
        for key in ("n_samples", "hist_bins"):
            if key in data:
                config[key] = int(data[key])
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate privacy loss report.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="JSON config path containing 'n_samples' and 'hist_bins'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # シード値を設定して再現性を確保（サンプリングモードのアルゴリズム用）
    np.random.seed(42)

    config = load_config(args.config)
    common_kwargs = {
        "n_samples": config["n_samples"],
        "hist_bins": config["hist_bins"],
    }

    input_sizes: Dict[str, int] = {
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
        "NoisyMaxSum": 20,
    }

    ideal_eps: Dict[str, Optional[float]] = {
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
        "NoisyMaxSum": float("inf"),
    }

    results: List[Tuple[str, int, float, float]] = []  # (name, size, eps, time)

    def timed_estimate(name, *args, **kwargs):
        """実行時間を計測しながらε推定を実行"""
        print(f"Estimating {name}...", end=" ", flush=True)
        start = time.time()
        eps_val = estimate_algorithm(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Done ({elapsed:.2f}s)")
        return eps_val, elapsed

    # ヒストグラム系
    hist_pairs = generate_hist_pairs(input_sizes["NoisyHist1"])
    eps_val, elapsed = timed_estimate("NoisyHist1", "NoisyHist1", hist_pairs, dist_func=_resolve_dist_func(noisy_hist1), **common_kwargs)
    results.append(("NoisyHist1", input_sizes["NoisyHist1"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("NoisyHist2", "NoisyHist2", hist_pairs, dist_func=_resolve_dist_func(noisy_hist2), **common_kwargs)
    results.append(("NoisyHist2", input_sizes["NoisyHist2"], eps_val, elapsed))

    # Report Noisy Max / Laplace 系
    vec_pairs = generate_change_one_pairs(input_sizes["ReportNoisyMax1"])

    eps_val, elapsed = timed_estimate("ReportNoisyMax1", "ReportNoisyMax1", vec_pairs, dist_func=_resolve_dist_func(report_noisy_max1), **common_kwargs)
    results.append(("ReportNoisyMax1", input_sizes["ReportNoisyMax1"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("ReportNoisyMax3", "ReportNoisyMax3", vec_pairs, dist_func=_resolve_dist_func(report_noisy_max3), **common_kwargs)
    results.append(("ReportNoisyMax3", input_sizes["ReportNoisyMax3"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("ReportNoisyMax2", "ReportNoisyMax2", vec_pairs, dist_func=_resolve_dist_func(report_noisy_max2), **common_kwargs)
    results.append(("ReportNoisyMax2", input_sizes["ReportNoisyMax2"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("ReportNoisyMax4", "ReportNoisyMax4", vec_pairs, dist_func=_resolve_dist_func(report_noisy_max4), **common_kwargs)
    results.append(("ReportNoisyMax4", input_sizes["ReportNoisyMax4"], eps_val, elapsed))

    laplace_pairs = generate_change_one_pairs(input_sizes["LaplaceMechanism"])

    eps_val, elapsed = timed_estimate("LaplaceMechanism", "LaplaceMechanism", laplace_pairs, dist_func=_resolve_dist_func(laplace_vec), **common_kwargs)
    results.append(("LaplaceMechanism", input_sizes["LaplaceMechanism"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate(
        "LaplaceParallel",
        "LaplaceParallel",
        [laplace_pairs[0]],
        dist_func=lambda data, eps: _resolve_dist_func(laplace_parallel)(data, 0.005, input_sizes["LaplaceParallel"]),
        **common_kwargs,
    )
    results.append(("LaplaceParallel", input_sizes["LaplaceParallel"], eps_val, elapsed))

    # SVT 系は algorithms ディレクトリの分布実装を使用
    svt_pairs_short = generate_change_one_pairs(input_sizes["SVT1"])
    svt_pairs_long = generate_change_one_pairs(input_sizes["SVT5"])

    eps_val, elapsed = timed_estimate("SVT1", "SVT1", svt_pairs_short, dist_func=_resolve_dist_func(svt1), **common_kwargs)
    results.append(("SVT1", input_sizes["SVT1"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("SVT2", "SVT2", svt_pairs_short, dist_func=_resolve_dist_func(svt2), **common_kwargs)
    results.append(("SVT2", input_sizes["SVT2"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("SVT3", "SVT3", svt_pairs_short, dist_func=_resolve_dist_func(svt3), **common_kwargs)
    results.append(("SVT3", input_sizes["SVT3"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("SVT4", "SVT4", svt_pairs_short, dist_func=_resolve_dist_func(svt4), **common_kwargs)
    results.append(("SVT4", input_sizes["SVT4"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("SVT5", "SVT5", svt_pairs_long, dist_func=_resolve_dist_func(svt5), **common_kwargs)
    results.append(("SVT5", input_sizes["SVT5"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate("SVT6", "SVT6", svt_pairs_long, dist_func=_resolve_dist_func(svt6), **common_kwargs)
    results.append(("SVT6", input_sizes["SVT6"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate(
        "NumericalSVT",
        "NumericalSVT",
        generate_change_one_pairs(input_sizes["NumericalSVT"]),
        dist_func=_resolve_dist_func(numerical_svt),
        **common_kwargs,
    )
    results.append(("NumericalSVT", input_sizes["NumericalSVT"], eps_val, elapsed))

    # 機構ベースの推定
    eps_val, elapsed = timed_estimate(
        "PrefixSum",
        "PrefixSum",
        generate_change_one_pairs(input_sizes["PrefixSum"]),
        mechanism=PrefixSum(eps=0.1),
        **common_kwargs,
    )
    results.append(("PrefixSum", input_sizes["PrefixSum"], eps_val, elapsed))

    eps_val, elapsed = timed_estimate(
        "SVT34Parallel",
        "SVT34Parallel",
        svt_pairs_long,
        mechanism=SVT34Parallel(eps=0.1),
        **common_kwargs,
    )
    results.append(("SVT34Parallel", input_sizes["SVT34Parallel"], eps_val, elapsed))

    otr_pairs = generate_change_one_pairs(input_sizes["OneTimeRAPPOR"])
    eps_val, elapsed = timed_estimate("OneTimeRAPPOR", "OneTimeRAPPOR", otr_pairs, dist_func=_resolve_dist_func(one_time_rappor), **common_kwargs)
    results.append(("OneTimeRAPPOR", input_sizes["OneTimeRAPPOR"], eps_val, elapsed))

    rappor_pairs = generate_change_one_pairs(input_sizes["RAPPOR"])
    eps_val, elapsed = timed_estimate("RAPPOR", "RAPPOR", rappor_pairs, dist_func=_resolve_dist_func(rappor), **common_kwargs)
    results.append(("RAPPOR", input_sizes["RAPPOR"], eps_val, elapsed))

    tg_pairs = [(np.array([2]), np.array([1])), (np.array([1]), np.array([0]))]
    eps_val, elapsed = timed_estimate(
        "TruncatedGeometric",
        "TruncatedGeometric",
        tg_pairs,
        mechanism=TruncatedGeometricMechanism(eps=0.1, n=5),
        **common_kwargs,
    )
    results.append(("TruncatedGeometric", input_sizes["TruncatedGeometric"], eps_val, elapsed))

    # NoisyMaxSum
    noisy_max_sum_pairs = generate_change_one_pairs(input_sizes["NoisyMaxSum"])
    eps_val, elapsed = timed_estimate("NoisyMaxSum", "NoisyMaxSum", noisy_max_sum_pairs, dist_func=_resolve_dist_func(noisy_max_sum), **common_kwargs)
    results.append(("NoisyMaxSum", input_sizes["NoisyMaxSum"], eps_val, elapsed))

    # レポート出力
    print("\n" + "="*70)
    print("Privacy Loss Report")
    print("="*70 + "\n")

    with open("docs/privacy_loss_report.md", "w") as f:
        f.write("# Privacy Loss Report\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **n_samples**: {config['n_samples']:,}\n")
        f.write(f"- **hist_bins**: {config['hist_bins']}\n")
        f.write(f"- **seed**: 42\n\n")
        f.write("## Results\n\n")
        f.write("| Algorithm | Input size | Estimated ε | Ideal ε | Time (s) |\n")
        f.write("|-----------|------------|-------------|---------|----------|\n")
        for name, size, eps_val, elapsed in results:
            ideal = ideal_eps.get(name)
            if ideal is None:
                ideal_str = "N/A"
            elif math.isinf(ideal):
                ideal_str = "∞"
            else:
                ideal_str = f"{ideal:.2f}"
            f.write(f"| {name} | {size} | {eps_val:.4f} | {ideal_str} | {elapsed:.2f} |\n")

    for name, size, eps_val, elapsed in results:
        ideal = ideal_eps.get(name)
        if ideal is None:
            print(f"{name} (n={size}): ε ≈ {eps_val:.4f}, time: {elapsed:.2f}s")
        else:
            ideal_display = "∞" if math.isinf(ideal) else f"{ideal:.2f}"
            print(f"{name} (n={size}): ε ≈ {eps_val:.4f} (ideal {ideal_display}), time: {elapsed:.2f}s")

    total_time = sum(elapsed for _, _, _, elapsed in results)
    print(f"\n{'='*70}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
