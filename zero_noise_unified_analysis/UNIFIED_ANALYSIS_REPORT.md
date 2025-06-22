
# UNIFIED DERIVATIVE BENCHMARK REPORT
Generated: 2025-06-19 20:31:39

## EXECUTIVE SUMMARY

This report analyzes performance data from all Python and Julia methods across multiple derivative orders and noise levels. The full, granular dataset is available in `RAW_MASTER_TABLE.csv`.

- **Implementations**: Python, Julia
- **Methods Compared**: 31 total methods
- **Derivatives Tested**: 0, 1, 2, 3, 4

---

## TOP PERFORMERS BY DERIVATIVE ORDER

Top 5 methods for each derivative order, based on average RMSE across all noise levels.


### Derivative Order 0

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | FiniteDiff_Python | 0.00e+00 |
| 2 | CubicSpline_Python | 1.37e-16 |
| 3 | RBF_ThinPlate_Python | 9.64e-13 |
| 4 | JuliaAAASmoothBary_Julia | 1.52e-10 |
| 5 | JuliaAAALS_Julia | 3.14e-09 |

### Derivative Order 1

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | AAA_Julia | 2.37e-06 |
| 2 | JuliaAAALS_Julia | 1.61e-05 |
| 3 | JuliaAAATwoStage_Julia | 1.61e-05 |
| 4 | JuliaAAASmoothBary_Julia | 2.35e-05 |
| 5 | JuliaAAAFullOpt_Julia | 2.54e-03 |

### Derivative Order 2

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | AAA_Julia | 1.32e-04 |
| 2 | JuliaAAALS_Julia | 2.77e-04 |
| 3 | JuliaAAATwoStage_Julia | 2.77e-04 |
| 4 | JuliaAAASmoothBary_Julia | 9.50e-04 |
| 5 | JuliaAAAFullOpt_Julia | 4.63e-02 |

### Derivative Order 3

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | JuliaAAATwoStage_Julia | 2.00e-02 |
| 2 | JuliaAAALS_Julia | 2.00e-02 |
| 3 | AAA_Julia | 9.38e-02 |
| 4 | JuliaAAAFullOpt_Julia | 3.94e+00 |
| 5 | JuliaAAASmoothBary_Julia | 7.25e+00 |

### Derivative Order 4

| Rank | Method | Avg RMSE |
|------|--------|----------|
| 1 | JuliaAAATwoStage_Julia | 6.94e-01 |
| 2 | JuliaAAALS_Julia | 6.94e-01 |
| 3 | AAA_Julia | 1.96e+02 |
| 4 | GPR_Julia | 3.20e+02 |
| 5 | SVR_Python | 4.24e+02 |

---

## FILES GENERATED

- `RAW_MASTER_TABLE.csv`: The complete, raw data from both benchmarks. **This is your single source of raw data.**
- `rmse_by_derivative_plots.png`: A plot showing RMSE vs. Noise Level for each derivative order.

## DATA SOURCES

- Python Raw Data: `results/python_raw_benchmark.csv`
- Julia Raw Data: `results/julia_raw_benchmark.csv`
