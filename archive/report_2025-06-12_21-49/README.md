# Derivative Approximation Benchmark Report

Generated on: 2025-06-12 21:50:03

## Executive Summary

This report analyzes the performance of **3 approximation methods** across **4 derivative orders** and **5 noise levels** using the Lotka-Volterra periodic system.

### Key Findings

#### 🏆 Best Methods by Derivative Order

| Derivative Order | Best Method | RMSE |
|-----------------|-------------|------|
| 0 | GPR | 2.65e-02 |
| 1 | GPR | 1.90e-01 |
| 2 | GPR | 2.13e+00 |
| 3 | GPR | 2.30e+01 |


#### 📊 Study Parameters
- **Methods Tested**: AAA, GPR, LOESS
- **Noise Levels**: 1.0e-04, 1.0e-03, 5.0e-03, 1.0e-02, 5.0e-02
- **Derivative Orders**: 0, 1, 2, 3
- **Data Size**: 101 points per experiment
- **Total Experiments**: 120 combinations

#### 🎯 Summary Statistics

**Overall Method Rankings** (by average RMSE across all conditions):
1. **GPR**: 6.33e+00
2. **AAA**: 1.73e+07
3. **LOESS**: 3.17e+07


#### 🔍 Key Insights

1. **Performance Degradation**: All methods show increasing RMSE with higher derivative orders
2. **Noise Sensitivity**: Performance varies significantly with noise level
3. **Method Specialization**: Different methods excel at different derivative orders

## Visualizations

- `figures/method_derivative_heatmap.png`: RMSE heatmap by method and derivative order
- `figures/noise_performance.png`: Performance vs noise level (log-log plots)
- `figures/method_rankings.png`: Overall method performance rankings
- `figures/derivative_degradation.png`: Performance degradation with derivative order

## Data Files

- `best_methods_per_derivative.csv`: Best performing method for each derivative order
- `comprehensive_summary_statistics.csv`: Detailed statistics for all method/derivative combinations

## Methodology

- **Benchmark System**: Lotka-Volterra periodic ODE system
- **Summary Statistic**: Root Mean Square Error (RMSE)
- **Evaluation**: Function values and derivatives (orders 0-5)
- **Cross-validation**: Multiple noise levels and observables tested

---

*Report generated using Python with pandas, matplotlib, and seaborn*
