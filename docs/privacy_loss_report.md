# Privacy Loss Report

## Configuration

- **n_samples**: 1,000,000
- **hist_bins**: 100
- **seed**: 42

## Results

| Algorithm | Input size | Estimated ε | Ideal ε | Time (s) | Method |
|-----------|------------|-------------|---------|----------|--------|
| NoisyHist1 | 5 | 0.5001 | 0.10 | 0.04 | 解析 |
| NoisyHist2 | 5 | 19.9901 | 10.00 | 0.03 | 解析 |
| ReportNoisyMax1 | 5 | 0.1040 | 0.10 | 2.72 | 解析 |
| ReportNoisyMax3 | 5 | 1.5520 | ∞ | 2.71 | 解析 |
| ReportNoisyMax2 | 5 | 0.0964 | 0.10 | 2.71 | 解析 |
| ReportNoisyMax4 | 5 | 8.6719 | ∞ | 2.72 | 解析 |
| LaplaceMechanism | 1 | 0.1000 | 0.10 | 0.01 | 解析 |
| LaplaceParallel | 20 | 0.1003 | 0.10 | 0.02 | 解析 |
| SVT1 | 10 | 0.0920 | 0.10 | 273.63 | サンプリング |
| SVT2 | 10 | 0.0843 | 0.10 | 282.52 | サンプリング |
| SVT3 | 10 | inf | ∞ | 46.03 | サンプリング |
| SVT4 | 10 | 0.1761 | 0.18 | 307.74 | サンプリング |
| SVT5 | 10 | inf | ∞ | 24.81 | サンプリング |
| SVT6 | 10 | 0.4976 | ∞ | 281.40 | サンプリング |
| NumericalSVT | 10 | inf | 0.10 | 58.76 | サンプリング |
| PrefixSum | 10 | inf | 0.10 | 125.87 | サンプリング |
| SVT34Parallel | 10 | inf | ∞ | 105.44 | サンプリング |
| OneTimeRAPPOR | 1 | 0.6005 | 0.80 | 0.00 | 解析 |
| RAPPOR | 1 | 0.3001 | 0.40 | 0.00 | 解析 |
| TruncatedGeometric | 5 | 0.1312 | 0.12 | 23.62 | サンプリング |
| NoisyMaxSum | 20 | 2.7707 | ∞ | 24.27 | 解析 |
