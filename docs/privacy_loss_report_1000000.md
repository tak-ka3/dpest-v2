# Privacy Loss Report

## Configuration

- **n_samples**: 1,000,000
- **hist_bins**: 100
- **seed**: 42

## Results

| Algorithm | Input size | Estimated ε | Ideal ε | Time (s) |
|-----------|------------|-------------|---------|----------|
| NoisyHist1 | 5 | 0.5001 | 0.10 | 0.04 |
| NoisyHist2 | 5 | 19.9901 | 10.00 | 0.03 |
| ReportNoisyMax1 | 5 | 0.1040 | 0.10 | 2.68 |
| ReportNoisyMax3 | 5 | 1.5520 | ∞ | 2.70 |
| ReportNoisyMax2 | 5 | 0.0964 | 0.10 | 2.67 |
| ReportNoisyMax4 | 5 | 8.6719 | ∞ | 2.68 |
| LaplaceMechanism | 1 | 0.1000 | 0.10 | 0.01 |
| LaplaceParallel | 20 | 0.1003 | 0.10 | 0.02 |
| SVT1 | 10 | 0.0920 | 0.10 | 272.33 |
| SVT2 | 10 | 0.0843 | 0.10 | 278.44 |
| SVT3 | 10 | 2.3026 | ∞ | 323.99 |
| SVT4 | 10 | 0.1730 | 0.18 | 307.77 |
| SVT5 | 10 | 0.5611 | ∞ | 0.02 |
| SVT6 | 10 | 0.2805 | ∞ | 0.06 |
| NumericalSVT | 10 | 2.8904 | 0.10 | 404.25 |
| PrefixSum | 10 | 0.6931 | 0.10 | 282.42 |
| SVT34Parallel | 10 | 2.8332 | ∞ | 154.07 |
| OneTimeRAPPOR | 1 | 0.6005 | 0.80 | 0.00 |
| RAPPOR | 1 | 0.3001 | 0.40 | 0.00 |
| TruncatedGeometric | 5 | 0.1267 | 0.12 | 0.14 |
| NoisyMaxSum | 20 | 2.7707 | ∞ | 23.87 |
