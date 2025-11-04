import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# このファイルを直接実行する際に dpest パッケージを読み込めるようにする
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dpest.analysis import estimate_algorithm, generate_change_one_pairs, generate_hist_pairs
from dpest.algorithms import (
    laplace_parallel_dist,
    laplace_vec_dist,
    noisy_hist1_dist,
    noisy_hist2_dist,
    numerical_svt_dist,
    one_time_rappor_dist,
    rappor_dist,
    report_noisy_max1_dist,
    report_noisy_max2_dist,
    report_noisy_max3_dist,
    report_noisy_max4_dist,
    svt1_dist,
    svt2_dist,
    svt3_dist,
    svt4_dist,
    svt5_dist,
    svt6_dist,
)
from dpest.mechanisms.geometric import TruncatedGeometricMechanism
from dpest.mechanisms.parallel import SVT34Parallel
from dpest.mechanisms.prefix_sum import PrefixSum


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
    }

    results: List[Tuple[str, int, float]] = []

    # ヒストグラム系
    hist_pairs = generate_hist_pairs(input_sizes["NoisyHist1"])
    results.append((
        "NoisyHist1",
        input_sizes["NoisyHist1"],
        estimate_algorithm("NoisyHist1", hist_pairs, dist_func=noisy_hist1_dist, **common_kwargs),
    ))
    results.append((
        "NoisyHist2",
        input_sizes["NoisyHist2"],
        estimate_algorithm("NoisyHist2", hist_pairs, dist_func=noisy_hist2_dist, **common_kwargs),
    ))

    # Report Noisy Max / Laplace 系
    vec_pairs = generate_change_one_pairs(input_sizes["ReportNoisyMax1"])
    results.append((
        "ReportNoisyMax1",
        input_sizes["ReportNoisyMax1"],
        estimate_algorithm("ReportNoisyMax1", vec_pairs, dist_func=report_noisy_max1_dist, **common_kwargs),
    ))
    results.append((
        "ReportNoisyMax3",
        input_sizes["ReportNoisyMax3"],
        estimate_algorithm("ReportNoisyMax3", vec_pairs, dist_func=report_noisy_max3_dist, **common_kwargs),
    ))
    results.append((
        "ReportNoisyMax2",
        input_sizes["ReportNoisyMax2"],
        estimate_algorithm("ReportNoisyMax2", vec_pairs, dist_func=report_noisy_max2_dist, **common_kwargs),
    ))
    results.append((
        "ReportNoisyMax4",
        input_sizes["ReportNoisyMax4"],
        estimate_algorithm("ReportNoisyMax4", vec_pairs, dist_func=report_noisy_max4_dist, **common_kwargs),
    ))

    laplace_pairs = generate_change_one_pairs(input_sizes["LaplaceMechanism"])
    results.append((
        "LaplaceMechanism",
        input_sizes["LaplaceMechanism"],
        estimate_algorithm("LaplaceMechanism", laplace_pairs, dist_func=laplace_vec_dist, **common_kwargs),
    ))
    results.append((
        "LaplaceParallel",
        input_sizes["LaplaceParallel"],
        estimate_algorithm(
            "LaplaceParallel",
            [laplace_pairs[0]],
            dist_func=lambda data, eps: laplace_parallel_dist(
                data, 0.005, input_sizes["LaplaceParallel"]
            ),
            **common_kwargs,
        ),
    ))

    # SVT 系は algorithms ディレクトリの分布実装を使用
    svt_pairs_short = generate_change_one_pairs(input_sizes["SVT1"])
    svt_pairs_long = generate_change_one_pairs(input_sizes["SVT5"])
    results.append((
        "SVT1",
        input_sizes["SVT1"],
        estimate_algorithm("SVT1", svt_pairs_short, dist_func=svt1_dist, **common_kwargs),
    ))
    results.append((
        "SVT2",
        input_sizes["SVT2"],
        estimate_algorithm("SVT2", svt_pairs_short, dist_func=svt2_dist, **common_kwargs),
    ))
    results.append((
        "SVT3",
        input_sizes["SVT3"],
        estimate_algorithm("SVT3", svt_pairs_short, dist_func=svt3_dist, **common_kwargs),
    ))
    results.append((
        "SVT4",
        input_sizes["SVT4"],
        estimate_algorithm("SVT4", svt_pairs_short, dist_func=svt4_dist, **common_kwargs),
    ))
    results.append((
        "SVT5",
        input_sizes["SVT5"],
        estimate_algorithm("SVT5", svt_pairs_long, dist_func=svt5_dist, **common_kwargs),
    ))
    results.append((
        "SVT6",
        input_sizes["SVT6"],
        estimate_algorithm("SVT6", svt_pairs_long, dist_func=svt6_dist, **common_kwargs),
    ))
    results.append((
        "NumericalSVT",
        input_sizes["NumericalSVT"],
        estimate_algorithm(
            "NumericalSVT",
            generate_change_one_pairs(input_sizes["NumericalSVT"]),
            dist_func=numerical_svt_dist,
            **common_kwargs,
        ),
    ))

    # 機構ベースの推定
    results.append((
        "PrefixSum",
        input_sizes["PrefixSum"],
        estimate_algorithm(
            "PrefixSum",
            generate_change_one_pairs(input_sizes["PrefixSum"]),
            mechanism=PrefixSum(eps=0.1),
            **common_kwargs,
        ),
    ))
    results.append((
        "SVT34Parallel",
        input_sizes["SVT34Parallel"],
        estimate_algorithm(
            "SVT34Parallel",
            svt_pairs_long,
            mechanism=SVT34Parallel(eps=0.1),
            **common_kwargs,
        ),
    ))

    otr_pairs = generate_change_one_pairs(input_sizes["OneTimeRAPPOR"])
    results.append((
        "OneTimeRAPPOR",
        input_sizes["OneTimeRAPPOR"],
        estimate_algorithm("OneTimeRAPPOR", otr_pairs, dist_func=one_time_rappor_dist, **common_kwargs),
    ))
    rappor_pairs = generate_change_one_pairs(input_sizes["RAPPOR"])
    results.append((
        "RAPPOR",
        input_sizes["RAPPOR"],
        estimate_algorithm("RAPPOR", rappor_pairs, dist_func=rappor_dist, **common_kwargs),
    ))

    tg_pairs = [(np.array([2]), np.array([1])), (np.array([1]), np.array([0]))]
    results.append((
        "TruncatedGeometric",
        input_sizes["TruncatedGeometric"],
        estimate_algorithm(
            "TruncatedGeometric",
            tg_pairs,
            mechanism=TruncatedGeometricMechanism(eps=0.1, n=5),
            **common_kwargs,
        ),
    ))

    # レポート出力
    with open("docs/privacy_loss_report.md", "w") as f:
        f.write("# Privacy Loss Report\n\n")
        f.write("| Algorithm | Input size | Estimated ε | Ideal ε |\n")
        f.write("|-----------|------------|-------------|---------|\n")
        for name, size, eps_val in results:
            ideal = ideal_eps.get(name)
            if ideal is None:
                ideal_str = "N/A"
            elif math.isinf(ideal):
                ideal_str = "∞"
            else:
                ideal_str = f"{ideal:.2f}"
            f.write(f"| {name} | {size} | {eps_val:.4f} | {ideal_str} |\n")

    for name, size, eps_val in results:
        ideal = ideal_eps.get(name)
        if ideal is None:
            print(f"{name} (n={size}): ε ≈ {eps_val:.4f}")
        else:
            ideal_display = "∞" if math.isinf(ideal) else f"{ideal:.2f}"
            print(f"{name} (n={size}): ε ≈ {eps_val:.4f} (ideal {ideal_display})")


if __name__ == "__main__":
    main()
