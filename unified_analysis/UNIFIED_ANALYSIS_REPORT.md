
# UNIFIED DERIVATIVE BENCHMARK REPORT
Generated: 2025-06-15 04:27:26

## EXECUTIVE SUMMARY

This report analyzes performance data from all Python and Julia methods across multiple derivative orders and noise levels. The full, granular dataset is available in `RAW_MASTER_TABLE.csv`.

- **Implementations**: Python, Julia
- **Methods Compared**: 25 total methods
- **Derivatives Tested**: 0, 1, 2, 3

---

## TOP PERFORMERS BY DERIVATIVE ORDER

Top 5 methods for each derivative order, based on average RMSE across all noise levels.


### Derivative Order 0

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | GP_Matern_Python | 2.64e-01 |
| 2 | GP_Matern_1.5_Python | 2.64e-01 |
| 3 | GP_Matern_2.5_Python | 2.70e-01 |
| 4 | GPR_Julia | 4.21e-01 |
| 5 | GP_RBF_Iso_Python | 4.69e-01 |

### Derivative Order 1

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | GP_Matern_2.5_Python | 1.91e-01 |
| 2 | GP_Matern_1.5_Python | 2.25e-01 |
| 3 | GP_Matern_Python | 2.25e-01 |
| 4 | GPR_Julia | 2.63e-01 |
| 5 | GP_RBF_Iso_Python | 2.88e-01 |

### Derivative Order 2

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | GP_Matern_2.5_Python | 2.74e+00 |
| 2 | GP_RBF_Iso_Python | 2.97e+00 |
| 3 | GP_RBF_Python | 2.97e+00 |
| 4 | AAA_LS_Python | 5.31e+00 |
| 5 | Butterworth_Python | 6.32e+00 |

### Derivative Order 3

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | GP_Matern_2.5_Python | 4.00e+01 |
| 2 | GP_RBF_Iso_Python | 4.54e+01 |
| 3 | GP_RBF_Python | 4.54e+01 |
| 4 | Butterworth_Python | 4.89e+01 |
| 5 | AAA_LS_Python | 1.86e+02 |

---

## FILES GENERATED

- `RAW_MASTER_TABLE.csv`: The complete, raw data from both benchmarks. **This is your single source of raw data.**
- `rmse_by_derivative_plots.png`: A plot showing RMSE vs. Noise Level for each derivative order.

## DATA SOURCES

- Python Raw Data: `results/python_raw_benchmark.csv`
- Julia Raw Data: `results/julia_raw_benchmark.csv`
