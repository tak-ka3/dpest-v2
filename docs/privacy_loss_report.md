# Privacy Loss Report

SVT1 と SVT2 の ε は依存する出力ベクトルを `svt1_joint_dist` や `svt2_joint_dist` で列挙したジョイント分布から評価しており、各座標を独立と仮定する場合よりも小さくなる。

| Algorithm | Input size | Estimated ε | Ideal ε |
|-----------|------------|-------------|---------|
| NoisyHist1 | 5 | 0.5001 | 0.10 |
| NoisyHist2 | 5 | 19.9901 | 10.00 |
| ReportNoisyMax1 | 5 | 0.1040 | 0.10 |
| ReportNoisyMax3 | 5 | 1.5520 | ∞ |
| LaplaceMechanism | 1 | 0.1000 | 0.10 |
| LaplaceParallel | 20 | 0.1003 | 0.10 |
| ReportNoisyMax2 | 5 | 0.0964 | 0.10 |
| ReportNoisyMax4 | 5 | 8.6719 | ∞ |
| SVT1 | 10 | 0.0969 | 0.10 |
| SVT2 | 10 | 0.2854 | 0.10 |
| SVT3 | 10 | 11.8574 | ∞ |
| SVT4 | 10 | 0.4904 | 0.18 |
| SVT5 | 10 | 0.5611 | ∞ |
| SVT6 | 10 | 0.2856 | ∞ |
| NumericalSVT | 10 | 16.9325 | 0.10 |
| PrefixSum | 10 | 17.2371 | 0.10 |
| OneTimeRAPPOR | 1 | 0.6005 | 0.80 |
| RAPPOR | 1 | 0.3001 | 0.40 |
| SVT34Parallel | 10 | 12.7443 | ∞ |
| TruncatedGeometric | 5 | 0.1533 | 0.12 |
