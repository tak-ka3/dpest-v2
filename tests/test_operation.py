"""
全操作のテストスクリプト

operationsディレクトリに移動した演算のテストを行います。
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from dpest.core import Dist
from dpest.operations import (
    max_distribution, min_distribution, argmax_distribution,
    add_distributions, affine_transform,
    prefix_sum_distributions, sampled_distribution
)
from dpest.noise import create_laplace_noise, Laplace
from dpest.utils.privacy import estimate_privacy_loss

# mechanisms モジュールを dpsniper.mechanisms として参照できるように設定
import types
import dpest.mechanisms as mechanisms
import dpest.mechanisms.abstract
dp_module = types.ModuleType('dpsniper')
dp_module.mechanisms = mechanisms
sys.modules['dpsniper'] = dp_module
sys.modules['dpsniper.mechanisms'] = mechanisms
sys.modules['dpsniper.mechanisms.abstract'] = dpest.mechanisms.abstract


def generate_patterns(n: int):
    """画像1に基づく入力パターンを長さnで生成"""
    ones = np.ones(n, dtype=int)
    patterns = {
        "one_above": (
            ones,
            np.concatenate(([2], np.ones(n - 1, dtype=int)))
        ),
        "one_below": (
            ones,
            np.concatenate(([0], np.ones(n - 1, dtype=int)))
        ),
        "one_above_rest_below": (
            ones,
            np.concatenate(([2], np.zeros(n - 1, dtype=int)))
        ),
        "one_below_rest_above": (
            ones,
            np.concatenate(([0], np.full(n - 1, 2, dtype=int)))
        ),
        "half_half": (
            ones,
            np.concatenate((np.zeros(n // 2, dtype=int), np.full(n - n // 2, 2, dtype=int)))
        ),
        "all_above_all_below": (
            ones,
            np.full(n, 2, dtype=int)
        ),
        "x_shape": (
            np.concatenate((np.ones(n // 2, dtype=int), np.zeros(n - n // 2, dtype=int))),
            np.concatenate((np.zeros(n // 2, dtype=int), np.ones(n - n // 2, dtype=int)))
        ),
    }
    return {k: (np.array(a), np.array(b)) for k, (a, b) in patterns.items()}


def test_argmax_operations():
    """Argmax演算のテスト"""
    print("=== Argmax演算テスト ===")

    patterns = generate_patterns(10)
    for name, (a, a_prime) in patterns.items():
        print(f"-- pattern: {name} --")
        dists_a = [Dist.deterministic(float(v)) for v in a]
        dists_ap = [Dist.deterministic(float(v)) for v in a_prime]
        res_a = argmax_distribution(dists_a)
        res_ap = argmax_distribution(dists_ap)
        print(f"  a  argmax: {[(int(v), p) for v, p in res_a.atoms]}")
        print(f"  a' argmax: {[(int(v), p) for v, p in res_ap.atoms]}")
        eps = estimate_privacy_loss(res_a, res_ap)
        print(f"  推定ε: {eps:.3f}")
        print()


def test_max_min_operations():
    """Max/Min演算のテスト"""
    print("=== Max/Min演算テスト ===")

    patterns = generate_patterns(10)
    for name, (a, a_prime) in patterns.items():
        print(f"-- pattern: {name} --")
        dists_a = [Dist.deterministic(float(v)) for v in a]
        dists_ap = [Dist.deterministic(float(v)) for v in a_prime]
        max_a = max_distribution(dists_a)
        max_ap = max_distribution(dists_ap)
        min_a = min_distribution(dists_a)
        min_ap = min_distribution(dists_ap)
        max_val_a = max_a.atoms[0][0] if max_a.atoms else None
        max_val_ap = max_ap.atoms[0][0] if max_ap.atoms else None
        min_val_a = min_a.atoms[0][0] if min_a.atoms else None
        min_val_ap = min_ap.atoms[0][0] if min_ap.atoms else None
        print(f"  a  max={max_val_a} min={min_val_a}")
        print(f"  a' max={max_val_ap} min={min_val_ap}")
        eps_max = estimate_privacy_loss(max_a, max_ap)
        eps_min = estimate_privacy_loss(min_a, min_ap)
        print(f"  max推定ε: {eps_max:.3f}")
        print(f"  min推定ε: {eps_min:.3f}")
        print()


def test_continuous_operations():
    """連続分布の演算テスト"""
    print("=== 連続分布演算テスト ===")

    # ラプラス分布を作成
    lap1 = create_laplace_noise(b=1.0)
    lap2 = create_laplace_noise(b=1.5)
    lap2_shift = Laplace(b=1.5, mu=1.0).to_dist()
    
    print(f"ラプラス分布1: 格子点数={len(lap1.density['x'])}, 総質量={lap1.total_mass():.6f}")
    print(f"ラプラス分布2: 格子点数={len(lap2.density['x'])}, 総質量={lap2.total_mass():.6f}")
    
    # Max計算
    max_result = max_distribution([lap1, lap2])
    max_result_shift = max_distribution([lap1, lap2_shift])
    print(f"Max結果: 格子点数={len(max_result.density['x'])}, 総質量={max_result.total_mass():.6f}")
    
    # 平均値の近似計算
    x_grid = max_result.density['x']
    f_grid = max_result.density['f']
    dx = max_result.density['dx']
    mean_approx = np.sum(x_grid * f_grid * dx)
    print(f"Max分布の近似平均: {mean_approx:.3f}")
    eps = estimate_privacy_loss(max_result, max_result_shift)
    print(f"Max演算の推定ε: {eps:.3f}")
    print()


def test_mixed_operations():
    """混合演算テスト（加法変換など）"""
    print("=== 混合演算テスト ===")
    
    # 確定値分布
    det_dist = Dist.deterministic(5.0)
    
    # ラプラス分布
    lap_dist = create_laplace_noise(b=1.0)
    
    print("入力:")
    print(f"  確定値分布: 値=5.0")
    print(f"  ラプラス分布: b=1.0")

    # 加法: 5 + Lap(1)
    add_result = add_distributions(det_dist, lap_dist)
    add_result_prime = add_distributions(Dist.deterministic(6.0), lap_dist)
    print(f"加法結果: 格子点数={len(add_result.density['x'])}, 総質量={add_result.total_mass():.6f}")
    eps_add = estimate_privacy_loss(add_result, add_result_prime)
    print(f"加法の推定ε: {eps_add:.3f}")

    # アフィン変換: 2*X + 3
    affine_result = affine_transform(lap_dist, a=2.0, b=3.0)
    affine_result_prime = affine_transform(lap_dist, a=2.0, b=4.0)
    print(f"アフィン変換結果: 格子点数={len(affine_result.density['x'])}, 総質量={affine_result.total_mass():.6f}")
    eps_affine = estimate_privacy_loss(affine_result, affine_result_prime)
    print(f"アフィン変換の推定ε: {eps_affine:.3f}")
    
    # 平均値の変化確認
    x_grid = affine_result.density['x']
    f_grid = affine_result.density['f']
    dx = affine_result.density['dx']
    mean_approx = np.sum(x_grid * f_grid * dx)
    print(f"アフィン変換後の近似平均: {mean_approx:.3f} (期待値: 3.0)")
    print()


def test_prefix_sum_operation():
    """PrefixSum演算のテスト"""
    print("=== PrefixSum演算テスト ===")

    patterns = generate_patterns(10)
    for name, (a, a_prime) in patterns.items():
        print(f"-- pattern: {name} --")
        dists_a = [Dist.deterministic(float(v)) for v in a]
        dists_ap = [Dist.deterministic(float(v)) for v in a_prime]
        res_a = prefix_sum_distributions(dists_a)
        res_ap = prefix_sum_distributions(dists_ap)
        vals_a = [d.atoms[0][0] if d.atoms else None for d in res_a]
        vals_ap = [d.atoms[0][0] if d.atoms else None for d in res_ap]
        print(f"  a  prefix_sum={vals_a}")
        print(f"  a' prefix_sum={vals_ap}")
        eps = [estimate_privacy_loss(pa, qa) for pa, qa in zip(res_a, res_ap)]
        print(f"  推定ε={ [f'{e:.3f}' for e in eps] }")
        print()


def test_dependent_add_operation():
    """依存した確率変数の加法演算テスト"""
    print("=== 依存加法演算テスト ===")
    cov1 = [[1.0, 0.8], [0.8, 1.0]]
    cov2 = [[1.0, 0.5], [0.5, 1.0]]
    samples1 = np.random.multivariate_normal([0.0, 0.0], cov1, size=2000)
    samples2 = np.random.multivariate_normal([0.0, 0.0], cov2, size=2000)
    result1 = add_distributions(Dist.deterministic(0.0), Dist.deterministic(0.0), joint_samples=samples1)
    result2 = add_distributions(Dist.deterministic(0.0), Dist.deterministic(0.0), joint_samples=samples2)
    mean1 = np.sum(result1.density['x'] * result1.density['f'] * result1.density['dx'])
    mean2 = np.sum(result2.density['x'] * result2.density['f'] * result2.density['dx'])
    print(f"  共分散0.8の平均: {mean1:.3f}")
    print(f"  共分散0.5の平均: {mean2:.3f}")
    eps = estimate_privacy_loss(result1, result2)
    print(f"  推定ε: {eps:.3f}")
    print()


def test_noisy_argmax_vs_noisy_max():
    """ノイズ付きargmax vs ノイズ付きmaxの比較"""
    print("=== ノイズ付きargmax vs max 比較 ===")
    
    # 入力データ
    input_values = [3.0, 1.0, 4.0, 1.0, 5.0]  # max=5, argmax=4
    b = 1.0
    
    print(f"入力データ: {input_values}")
    print(f"真のmax: {max(input_values)}")
    print(f"真のargmax: {input_values.index(max(input_values))}")
    
    # 確定値分布を作成
    input_dists = [Dist.deterministic(val) for val in input_values]
    input_values_prime = input_values.copy()
    input_values_prime[0] += 1.0
    input_dists_prime = [Dist.deterministic(val) for val in input_values_prime]

    # ノイズ分布を作成
    noise_dists = create_laplace_noise(b=b, size=len(input_values))

    # ノイズ付き分布: x + noise
    noisy_dists = []
    noisy_dists_prime = []
    for x_dist, x_dist_p, noise_dist in zip(input_dists, input_dists_prime, noise_dists):
        noisy_dists.append(add_distributions(x_dist, noise_dist))
        noisy_dists_prime.append(add_distributions(x_dist_p, noise_dist))

    # Argmax計算
    argmax_result = argmax_distribution(noisy_dists)
    argmax_result_prime = argmax_distribution(noisy_dists_prime)
    print(f"ノイズ付きargmax分布: {[(int(v), f'{p:.3f}') for v, p in argmax_result.atoms]}")
    eps_argmax = estimate_privacy_loss(argmax_result, argmax_result_prime)
    print(f"  argmax推定ε: {eps_argmax:.3f}")

    # Max計算
    max_result = max_distribution(noisy_dists)
    max_result_prime = max_distribution(noisy_dists_prime)
    x_grid = max_result.density['x']
    f_grid = max_result.density['f']
    dx = max_result.density['dx']
    mean_max = np.sum(x_grid * f_grid * dx)
    print(f"ノイズ付きmax分布: 近似平均={mean_max:.3f}")
    eps_max = estimate_privacy_loss(max_result, max_result_prime)
    print(f"  max推定ε: {eps_max:.3f}")
    print()


def test_sampled_mechanism_operation():
    """Sampled演算を用いた機構の分布近似テスト"""
    print("=== Sampled演算による機構分布近似 ===")

    from dpest.mechanisms.report_noisy_max import ReportNoisyMax1
    from dpest.mechanisms.sparse_vector_technique import SparseVectorTechnique1

    # ReportNoisyMax1 の分布近似
    rnm = ReportNoisyMax1(eps=0.5)
    a = np.array([0.0, 1.0, 2.0])
    a_prime = a.copy()
    a_prime[0] += 1.0

    dist = sampled_distribution(lambda n: rnm.m(a, n_samples=n), n_samples=2000)
    dist_prime = sampled_distribution(lambda n: rnm.m(a_prime, n_samples=n), n_samples=2000)
    print(f"ReportNoisyMax1 出力分布: {dist.atoms}")
    print(f"  総質量: {dist.total_mass():.3f}")
    eps_rnm = estimate_privacy_loss(dist, dist_prime)
    print(f"  推定ε: {eps_rnm:.3f}")

    # SparseVectorTechnique1 の各クエリ分布
    svt = SparseVectorTechnique1(eps=0.2, c=1, t=0.5)
    a2 = np.array([0.0, 1.0, 2.0])
    a2_prime = a2.copy()
    a2_prime[0] += 1.0

    dists_a = sampled_distribution(lambda n: svt.m(a2, n_samples=n), n_samples=2000)
    dists_ap = sampled_distribution(lambda n: svt.m(a2_prime, n_samples=n), n_samples=2000)
    for idx, (d_a, d_ap) in enumerate(zip(dists_a, dists_ap)):
        eps = estimate_privacy_loss(d_a, d_ap)
        print(f"  SVT query{idx} 総質量: {d_a.total_mass():.3f} ε≈{eps:.3f}")
    print()


def test_svt_conditional_operation():
    """条件演算を用いたSparseVectorTechniqueの分布計算テスト"""
    print("=== SparseVectorTechnique 条件演算テスト ===")

    from dpest.mechanisms.sparse_vector_technique import SparseVectorTechnique5
    # 画像1のパターンを長さ10で生成
    patterns = generate_patterns(10)

    # 画像2の推奨値に基づき eps=0.1 を使用
    svt = SparseVectorTechnique5(eps=0.1, t=1.0)

    for name, (a, a_prime) in patterns.items():
        print(f"-- pattern: {name} --")
        dists_a = svt.dist(a)
        dists_ap = svt.dist(a_prime)
        masses_a = [d.total_mass() for d in dists_a]
        masses_ap = [d.total_mass() for d in dists_ap]
        print(f"  a ={list(a)} masses={[f'{m:.3f}' for m in masses_a]}")
        print(f"  a'={list(a_prime)} masses={[f'{m:.3f}' for m in masses_ap]}")
        eps = [estimate_privacy_loss(p, q) for p, q in zip(dists_a, dists_ap)]
        print(f"  推定ε={ [f'{e:.3f}' for e in eps] }")
        print()


def main():
    """メイン実行"""
    print("Operations ディレクトリ演算テスト")
    print("=" * 50)
    
    try:
        test_argmax_operations()
        test_max_min_operations()
        test_continuous_operations()
        test_mixed_operations()
        test_prefix_sum_operation()
        test_dependent_add_operation()
        test_noisy_argmax_vs_noisy_max()
        test_sampled_mechanism_operation()
        test_svt_conditional_operation()
        
        print("=" * 50)
        print("✅ 全てのテストが正常に完了しました")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

