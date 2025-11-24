# Privacy Loss Report

## Configuration

- **n_samples**: 100,000
- **hist_bins**: 100
- **seed**: 42

## Results

| Algorithm | Input size | Estimated ε | Ideal ε | Time (s) |
|-----------|------------|-------------|---------|----------|
| NoisyHist1 | 5 | 0.5001 | 0.10 | 0.04 |
| NoisyHist2 | 5 | 19.9901 | 10.00 | 0.03 |
| ReportNoisyMax1 | 5 | 0.1040 | 0.10 | 2.69 |
| ReportNoisyMax3 | 5 | 1.5520 | ∞ | 2.70 |
| ReportNoisyMax2 | 5 | 0.0964 | 0.10 | 2.68 |
| ReportNoisyMax4 | 5 | 8.6719 | ∞ | 2.68 |
| LaplaceMechanism | 1 | 0.1000 | 0.10 | 0.01 |
| LaplaceParallel | 20 | 0.1003 | 0.10 | 0.02 |
| SVT1 | 10 | 0.1393 | 0.10 | 27.61 |
| SVT2 | 10 | 0.1559 | 0.10 | 28.31 |
| SVT3 | 10 | 2.6391 | ∞ | 32.63 |
| SVT4 | 10 | 0.2024 | 0.18 | 31.53 |
| SVT5 | 10 | 0.5611 | ∞ | 0.02 |
| SVT6 | 10 | 0.2805 | ∞ | 0.06 |
| NumericalSVT | 10 | 2.7726 | 0.10 | 41.61 |
| PrefixSum | 10 | inf | 0.10 | 27.96 |
| SVT34Parallel | 10 | 2.9444 | ∞ | 15.43 |
| OneTimeRAPPOR | 1 | 0.6005 | 0.80 | 0.00 |
| RAPPOR | 1 | 0.3001 | 0.40 | 0.00 |
| TruncatedGeometric | 5 | 0.1527 | 0.12 | 0.02 |
| NoisyMaxSum | 20 | 2.7707 | ∞ | 24.31 |
