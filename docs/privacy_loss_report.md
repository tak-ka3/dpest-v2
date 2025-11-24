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
| ReportNoisyMax1 | 5 | 0.1040 | 0.10 | 2.73 |
| ReportNoisyMax3 | 5 | 1.5520 | ∞ | 2.69 |
| ReportNoisyMax2 | 5 | 0.0964 | 0.10 | 2.70 |
| ReportNoisyMax4 | 5 | 8.6719 | ∞ | 2.72 |
| LaplaceMechanism | 1 | 0.1000 | 0.10 | 0.01 |
| LaplaceParallel | 20 | 0.1003 | 0.10 | 0.02 |
| SVT1 | 10 | 0.0920 | 0.10 | 275.83 |
| SVT2 | 10 | 0.0843 | 0.10 | 282.64 |
| SVT3 | 10 | inf | ∞ | 328.03 |
| SVT4 | 10 | 0.1730 | 0.18 | 309.53 |
| SVT5 | 10 | inf | ∞ | 171.04 |
| SVT6 | 10 | 0.4949 | ∞ | 281.76 |
| NumericalSVT | 10 | inf | 0.10 | 407.91 |
| PrefixSum | 10 | inf | 0.10 | 275.09 |
| SVT34Parallel | 10 | inf | ∞ | 153.95 |
| OneTimeRAPPOR | 1 | 0.6005 | 0.80 | 0.01 |
| RAPPOR | 1 | 0.3001 | 0.40 | 0.01 |
| TruncatedGeometric | 5 | 0.1213 | 0.12 | 0.15 |
| NoisyMaxSum | 20 | 2.7707 | ∞ | 24.37 |
