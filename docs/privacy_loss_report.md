# Privacy Loss Report

## Configuration

- **n_samples**: 1,000,000
- **hist_bins**: 100
- **seed**: 42

## Results

| Algorithm | Input size | Estimated ε | Ideal ε | Time (s) | Method |
|-----------|------------|-------------|---------|----------|--------|
| NoisyHist1 | 5 | 0.1002 | 0.10 | 0.01 | 解析 |
| NoisyHist2 | 5 | 10.0001 | 10.00 | 0.01 | 解析 |
| ReportNoisyMax1 | 5 | 0.1069 | 0.10 | 2.75 | 解析 |
| ReportNoisyMax3 | 5 | 0.8477 | ∞ | 2.73 | 解析 |
| ReportNoisyMax2 | 5 | 0.0964 | 0.10 | 2.75 | 解析 |
| ReportNoisyMax4 | 5 | 8.6719 | ∞ | 2.74 | 解析 |
| LaplaceMechanism | 1 | 0.1002 | 0.10 | 0.01 | 解析 |
| LaplaceParallel | 20 | 0.1010 | 0.10 | 0.04 | 解析 |
| SVT1 | 10 | 0.0920 | 0.10 | 276.45 | サンプリング |
| SVT2 | 10 | 0.0843 | 0.10 | 282.88 | サンプリング |
| SVT3 | 10 | inf | ∞ | 47.02 | サンプリング |
| SVT4 | 10 | 0.1761 | 0.18 | 310.38 | サンプリング |
| SVT5 | 10 | inf | ∞ | 24.79 | サンプリング |
| SVT6 | 10 | 0.4976 | ∞ | 278.91 | サンプリング |
| NumericalSVT | 10 | inf | 0.10 | 57.76 | サンプリング |
| PrefixSum | 10 | inf | 0.10 | 123.84 | サンプリング |
| SVT34Parallel | 10 | inf | ∞ | 105.22 | サンプリング |
| OneTimeRAPPOR | 1 | 0.6005 | 0.80 | 0.01 | 解析 |
| RAPPOR | 1 | 0.3001 | 0.40 | 0.00 | 解析 |
| TruncatedGeometric | 5 | 0.1312 | 0.12 | 23.92 | サンプリング |
| NoisyMaxSum | 20 | 1.8692 | ∞ | 24.40 | 解析 |
