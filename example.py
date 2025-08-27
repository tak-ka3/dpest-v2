"""
使用例: ラプラスノイズを加えたargmax

DESIGN.mdで示された使用例を実装します。
"""

import numpy as np
from typing import List

# 直接インポート
from core import Dist
from engine import AlgorithmBuilder, Laplace_dist, vector_argmax


def noisy_argmax_lap(x: List[float], b: float = 1.0) -> Dist:
    """
    ラプラスノイズを加えたargmax
    
    Args:
        x: 入力ベクトル
        b: ラプラスノイズのスケールパラメータ
    
    Returns:
        argmaxの分布
    """
    # 入力を確定値分布のリストに変換
    x_dists = [Dist.deterministic(val) for val in x]
    
    # 各要素にラプラスノイズを追加
    noise_dists = Laplace_dist(b=b, size=len(x))
    
    # ベクトル加法: z = x + noise
    z_dists = AlgorithmBuilder.vector_add(x_dists, noise_dists)
    
    # argmaxを計算
    return vector_argmax(z_dists)


def estimate_eps_simple(P: Dist, Q: Dist) -> float:
    """
    簡単なε推定（実装例）
    
    実際のε推定はより複雑ですが、ここでは動作確認用の簡易版
    """
    if not P.atoms or not Q.atoms:
        return float('inf')  # 連続分布の場合は無限大
    
    # 離散分布の場合のKLダイバージェンス的な計算
    max_ratio = 0.0
    
    for p_val, p_prob in P.atoms:
        # Qでの対応する確率を見つける
        # TODO: より効率的な方法で対応を見つける
        if p_prob <= 0:
            continue
        q_prob = 0.0
        for q_val, q_p in Q.atoms:
            if abs(p_val - q_val) < 1e-10:  # 同じ値
                q_prob = q_p
                break
        
        if q_prob > 0:
            # ratio = p_prob / q_prob
            max_ratio = max(max_ratio, p_prob / q_prob, q_prob / p_prob)
    
    return np.log(max_ratio) if max_ratio > 0 else float('inf')


def main():
    """メイン実行例"""
    print("差分プライバシーε推定ライブラリの使用例")
    print("=" * 50)
    
    # テストデータセット
    all_datasets = [
        ([1, 1, 1, 1, 1], [0, 1, 1, 1, 1]),
        ([1, 1, 1, 1, 1], [2, 0, 0, 0, 0]),
        ([1, 1, 1, 1, 1], [0, 2, 2, 2, 2]),
        ([1, 1, 1, 1, 1], [0, 0, 0, 2, 2]),
        ([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]),
        ([1, 1, 0, 0, 0], [0, 0, 1, 1, 1]),
    ]
    
    # パラメータ
    b = 0.5  # ラプラスノイズのスケール。 2/b が理論的なε値。
    
    print(f"ラプラスノイズパラメータ b = {b}")
    print()
    
    for i, (D, D_prime) in enumerate(all_datasets):
        print(f"データセットペア {i+1}:")
        print(f"  D = {D}")
        print(f"  D' = {D_prime}")
        
        try:
            # 分布を計算
            P = noisy_argmax_lap(D, b=b)
            Q = noisy_argmax_lap(D_prime, b=b)
            
            print(f"  P (D での出力分布): {P}")
            print(f"  Q (D' での出力分布): {Q}")
            
            # ε値を推定
            eps_estimate = estimate_eps_simple(P, Q)
            print(f"  推定ε値: {eps_estimate:.4f}")
            
        except Exception as e:
            print(f"  エラー: {e}")
        
        print()


if __name__ == "__main__":
    main()