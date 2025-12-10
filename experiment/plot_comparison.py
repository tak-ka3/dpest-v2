#!/usr/bin/env python3
"""
Compare DPEST, DP-Sniper, and StatDP results
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV files
def read_csv(filename):
    data = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mechanism = row['mechanism']
            data[mechanism] = {
                'eps': float(row['eps']),
                'time': float(row['time'])
            }
    return data

dpsniper = read_csv('dpsniper.csv')
statdp = read_csv('stadp.csv')

# DPEST data from privacy_loss_report.md
dpest_data = {
    'LaplaceMechanism': {'eps': 0.1002, 'time': 0.01},
    'NoisyHist1': {'eps': 0.1002, 'time': 0.01},
    'NoisyHist2': {'eps': 10.0001, 'time': 0.01},
    'LaplaceParallel': {'eps': 0.1010, 'time': 0.04},
    'ReportNoisyMax1': {'eps': 0.1069, 'time': 2.75},
    'ReportNoisyMax2': {'eps': 0.0964, 'time': 2.75},
    'ReportNoisyMax3': {'eps': 0.8477, 'time': 2.73},
    'ReportNoisyMax4': {'eps': 8.6719, 'time': 2.74},
    'SparseVectorTechnique1': {'eps': 0.0920, 'time': 276.45},
    'SparseVectorTechnique2': {'eps': 0.0843, 'time': 282.88},
    'SparseVectorTechnique3': {'eps': float('inf'), 'time': 47.02},
    'SparseVectorTechnique4': {'eps': 0.1761, 'time': 310.38},
    'SparseVectorTechnique5': {'eps': float('inf'), 'time': 24.79},
    'SparseVectorTechnique6': {'eps': 0.4976, 'time': 278.91},
    'SVT34Parallel': {'eps': float('inf'), 'time': 105.22},
    'NumericalSVT': {'eps': float('inf'), 'time': 57.76},
    'PrefixSum': {'eps': float('inf'), 'time': 123.84},
    'OneTimeRAPPOR': {'eps': 0.6005, 'time': 0.01},
    'Rappor': {'eps': 0.3001, 'time': 0.001},
    'TruncatedGeometric': {'eps': 0.1312, 'time': 23.92},
}

# --- 理論値 ε (Ideal ε) の定義 ---
ideal_eps_data = {
    'NoisyHist1': 0.10,
    'NoisyHist2': 10.00,
    'ReportNoisyMax1': 0.10,
    'ReportNoisyMax3': float('inf'),
    'ReportNoisyMax2': 0.10,
    'ReportNoisyMax4': float('inf'),
    'LaplaceMechanism': 0.10,
    'LaplaceParallel': 0.10,
    'SparseVectorTechnique1': 0.10,   # SVT1
    'SparseVectorTechnique2': 0.10,   # SVT2
    'SparseVectorTechnique3': float('inf'),  # SVT3
    'SparseVectorTechnique4': 0.18,   # SVT4
    'SparseVectorTechnique5': float('inf'),  # SVT5
    'SparseVectorTechnique6': float('inf'),  # SVT6
    'NumericalSVT': 0.10,
    'PrefixSum': 0.10,
    'SVT34Parallel': float('inf'),
    'OneTimeRAPPOR': 0.80,
    'Rappor': 0.40,
    'TruncatedGeometric': 0.12,
    # 'NoisyMaxSum': float('inf'),  # 今回のプロットには含まれていない
}

# Map algorithm names (for display only)
name_mapping = {
    'SparseVectorTechnique1': 'SVT1',
    'SparseVectorTechnique2': 'SVT2',
    'SparseVectorTechnique3': 'SVT3',
    'SparseVectorTechnique4': 'SVT4',
    'SparseVectorTechnique5': 'SVT5',
    'SparseVectorTechnique6': 'SVT6',
}

# Select common algorithms (plot order, bottom → top)
common_algorithms = [
    'TruncatedGeometric',
    'Rappor',
    'OneTimeRAPPOR',
    'PrefixSum',
    'NumericalSVT',
    'SVT34Parallel',
    'SparseVectorTechnique6',
    'SparseVectorTechnique5',
    'SparseVectorTechnique4',
    'SparseVectorTechnique3',
    'SparseVectorTechnique2',
    'SparseVectorTechnique1',
    'ReportNoisyMax4',
    'ReportNoisyMax3',
    'ReportNoisyMax2',
    'ReportNoisyMax1',
    'LaplaceParallel',
    'NoisyHist2',
    'NoisyHist1',
    'LaplaceMechanism',
]

# Prepare data arrays
algorithms = []
dpest_eps = []
dpsniper_eps = []
statdp_eps = []
dpest_time = []
dpsniper_time = []
statdp_time = []
ideal_eps = []   # 理論値 ε をここに入れる

for algo in common_algorithms:
    display_name = name_mapping.get(algo, algo)
    algorithms.append(display_name)

    # DPEST
    dpest_eps.append(dpest_data[algo]['eps'])
    dpest_time.append(dpest_data[algo]['time'])

    # DP-Sniper
    if algo in dpsniper:
        dpsniper_eps.append(dpsniper[algo]['eps'])
        dpsniper_time.append(dpsniper[algo]['time'])
    else:
        dpsniper_eps.append(np.nan)
        dpsniper_time.append(np.nan)

    # StatDP
    if algo in statdp:
        statdp_eps.append(statdp[algo]['eps'])
        statdp_time.append(statdp[algo]['time'])
    else:
        statdp_eps.append(np.nan)
        statdp_time.append(np.nan)

    # Ideal ε
    ideal = ideal_eps_data.get(algo, np.nan)
    ideal_eps.append(ideal)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

y_pos = np.arange(len(algorithms))
bar_height = 0.2

colors = {
    'dpest': '#4CAF50',
    'dpsniper': '#2196F3',
    'statdp': '#9C27B0',
}

# ===== ε-plot (Left) =====
ax1.set_xlabel('Estimated ε', fontsize=12)
ax1.set_title('Privacy Loss (ε) Comparison', fontsize=14, fontweight='bold')

# ===== 無限大の推定値を xmax*0.98 に変換 =====
def replace_inf_with_xmax(values, xmax):
    return [
        (xmax * 0.98 if np.isinf(v) else v)
        for v in values
    ]

# xmax を取得（理論値も含む）
all_eps_values = [
    v for v in (dpest_eps + dpsniper_eps + statdp_eps + ideal_eps)
    if not np.isnan(v) and not np.isinf(v)
]

if all_eps_values:
    xmax = max(all_eps_values) * 1.1
else:
    xmax = 1.0

# ∞ のところを xmax*0.98 に置き換える
dpest_eps_plot = replace_inf_with_xmax(dpest_eps, xmax)
dpsniper_eps_plot = replace_inf_with_xmax(dpsniper_eps, xmax)
statdp_eps_plot = replace_inf_with_xmax(statdp_eps, xmax)


# StatDP → DP-Sniper → DPEST の順にバーを描画
bars_statdp = ax1.barh(y_pos - bar_height, statdp_eps_plot, bar_height,
                       label='StatDP', color=colors['statdp'])
bars_dpsniper = ax1.barh(y_pos, dpsniper_eps_plot, bar_height,
                         label='DP-Sniper', color=colors['dpsniper'])
bars_dpest = ax1.barh(y_pos + bar_height, dpest_eps_plot, bar_height,
                      label='DPEST', color=colors['dpest'])

# 値ラベル
for i, (v1, v2, v3) in enumerate(zip(statdp_eps, dpsniper_eps, dpest_eps)):

    # StatDP
    if np.isinf(v1):
        ax1.text(xmax*0.98, i - bar_height, 'inf', va='center', fontsize=8)
    elif not np.isnan(v1):
        ax1.text(v1, i - bar_height, f'{v1:.3f}', va='center', fontsize=8)

    # DP-Sniper
    if np.isinf(v2):
        ax1.text(xmax*0.98, i, 'inf', va='center', fontsize=8)
    elif not np.isnan(v2):
        ax1.text(v2, i, f'{v2:.3f}', va='center', fontsize=8)

    # DPEST
    if np.isinf(v3):
        ax1.text(xmax*0.98, i + bar_height, 'inf', va='center', fontsize=8, color='red')
    elif not np.isnan(v3):
        ax1.text(v3, i + bar_height, f'{v3:.3f}', va='center', fontsize=8, color='red')

# ★ 理論値 ε の縦線 + 数値ラベル (∞も含む) を追加 ★
added_ideal_label = False

# x 軸の最大値を取得（∞対策で後から使う）
all_eps_values = [
    v for v in (dpest_eps + dpsniper_eps + statdp_eps + ideal_eps)
    if not np.isnan(v) and not np.isinf(v)
]
if all_eps_values:
    xmax = max(all_eps_values) * 1.1
else:
    xmax = 1.0

for i, ideal in enumerate(ideal_eps):

    # -------------------------
    # ❶ ideal が有限の場合
    # -------------------------
    if not np.isnan(ideal) and not np.isinf(ideal):

        ax1.vlines(
            ideal,
            i - 0.4, i + 0.4,
            colors='black',
            linestyles='dashed',
            linewidth=1,
            label='Ideal ε' if not added_ideal_label else None
        )

        ax1.text(
            ideal,
            i + 0.45,
            f'{ideal:.2f}',
            ha='center',
            va='bottom',
            fontsize=7,
            color='blue'
        )

    # -------------------------
    # ❷ ideal が ∞ の場合
    # -------------------------
    elif np.isinf(ideal):

        inf_x = xmax * 0.98   # 右端の 98% 位置に描画

        # 縦線
        ax1.vlines(
            inf_x,
            i - 0.4, i + 0.4,
            colors='black',
            linestyles='dotted',
            linewidth=1,
            label='Ideal ε' if not added_ideal_label else None
        )

        # ラベル
        ax1.text(
            inf_x,
            i + 0.45,
            "inf",
            ha='center',
            va='bottom',
            fontsize=7,
            color='blue'
        )

    added_ideal_label = True


ax1.set_yticks(y_pos)
ax1.set_yticklabels(algorithms)

# x軸の右端をデータと理論値の最大に合わせて少し余裕を持たせる
all_eps_values = [
    v for v in (dpest_eps + dpsniper_eps + statdp_eps + ideal_eps)
    if not np.isnan(v) and not np.isinf(v)
]
if all_eps_values:
    xmax = max(all_eps_values) * 1.1
    ax1.set_xlim(left=0, right=xmax)

ax1.legend(loc='upper right')
ax1.grid(axis='x', alpha=0.3)

# ===== Time plot (Right) =====
ax2.set_xlabel('Execution Time (seconds)', fontsize=12)
ax2.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')

bars_statdp_t = ax2.barh(y_pos - bar_height, statdp_time, bar_height,
                         label='StatDP', color=colors['statdp'])
bars_dpsniper_t = ax2.barh(y_pos, dpsniper_time, bar_height,
                           label='DP-Sniper', color=colors['dpsniper'])
bars_dpest_t = ax2.barh(y_pos + bar_height, dpest_time, bar_height,
                        label='DPEST', color=colors['dpest'])

def format_time(sec):
    if sec < 1:
        return f'{sec:.2f}sec'
    elif sec < 60:
        return f'{sec:.0f}sec'
    else:
        return f'{sec/60:.0f}min'

for i, (t1, t2, t3) in enumerate(zip(statdp_time, dpsniper_time, dpest_time)):
    if not np.isnan(t1):
        ax2.text(t1, i - bar_height, format_time(t1), va='center', fontsize=8)
    if not np.isnan(t2):
        ax2.text(t2, i, format_time(t2), va='center', fontsize=8)
    if not np.isnan(t3):
        ax2.text(t3, i + bar_height, format_time(t3), va='center', fontsize=8, color='red')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(algorithms)
ax2.set_xscale('log')
ax2.legend(loc='upper right')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
print('Saved comparison_plot.png')
