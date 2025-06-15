# Derivative Approximation Benchmark

A comprehensive benchmark suite for comparing derivative approximation methods across different programming languages, noise levels, and derivative orders.

## Overview

This project benchmarks various methods for approximating derivatives of noisy time series data, with a focus on:
- Function approximation accuracy
- Derivative approximation at multiple orders (1st, 2nd, 3rd)
- Robustness to noise
- Performance across different implementations (Python vs Julia)

## Project Structure

```
derivative_approximation_benchmark/
├── README.md                           # This file
├── comprehensive_methods_library.py    # Core Python methods library
├── enhanced_gp_methods.py             # Enhanced Gaussian Process variants
├── run_proper_noisy_benchmark.py      # Main Python benchmark
├── run_gp_comparison.py               # GP kernel comparison
├── create_unified_comparison.py       # Unified analysis across all methods
├── results/                           # Benchmark results directory
└── unified_analysis/                  # Final analysis outputs
```

## Core Components

### 1. Methods Library (`comprehensive_methods_library.py`)

**Purpose**: Central library containing 14 derivative approximation methods across multiple categories.

**Categories**:
- **Interpolation**: CubicSpline, SmoothingSpline, RBF (Thin Plate, Multiquadric)
- **Gaussian Process**: GP with RBF kernel, GP with Matérn kernel
- **Polynomial**: Chebyshev polynomials, standard polynomial fitting
- **Smoothing**: Savitzky-Golay filter, Butterworth filter
- **Machine Learning**: Random Forest, Support Vector Regression
- **Spectral**: Fourier series approximation
- **Finite Difference**: Central difference schemes

**Key Features**:
- Standardized interface via `DerivativeApproximator` base class
- Automatic error handling and timing
- Support for derivatives up to 3rd order
- Consistent evaluation framework

**Usage**:
```python
from comprehensive_methods_library import create_all_methods

# Create all methods for given data
methods = create_all_methods(t, y)

# Evaluate a specific method
results = methods['CubicSpline'].evaluate(t_eval, max_derivative=3)
print(f"RMSE: {results['rmse']}, Success: {results['success']}")
```

### 2. Enhanced GP Methods (`enhanced_gp_methods.py`)

**Purpose**: Specialized Gaussian Process implementations with different kernels for detailed GP comparison.

**Kernel Types**:
- **RBF Isotropic**: Standard RBF kernel
- **Matérn (ν=0.5, 1.5, 2.5, 5.0)**: Different smoothness parameters
- **Rational Quadratic**: Mixture of RBF scales
- **Periodic**: For periodic data like ODE solutions

**Usage**:
```python
from enhanced_gp_methods import create_enhanced_gp_methods

# Create all GP variants
gp_methods = create_enhanced_gp_methods(t, y)

# Test specific kernel
results = gp_methods['GP_RBF_Iso'].evaluate(t_eval, max_derivative=2)
```

## Benchmark Scripts

### 3. Main Python Benchmark (`run_proper_noisy_benchmark.py`)

**Purpose**: Primary benchmark testing all Python methods on properly generated noisy data.

**What it does**:
1. Loads clean ODE solution data from Julia results
2. Adds controlled Gaussian noise at multiple levels: [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
3. Tests all 14 Python methods on noisy data
4. Evaluates performance against clean truth
5. Generates comprehensive performance analysis

**Key Features**:
- **Proper noise handling**: Fits methods to noisy data, evaluates against clean truth
- **Multiple derivative orders**: Tests function + 1st/2nd/3rd derivatives
- **Performance metrics**: RMSE, MAE, max error, timing
- **Robustness analysis**: Noise degradation ratios

**Usage**:
```bash
python run_proper_noisy_benchmark.py
# Output: results/proper_noisy_benchmark_YYYYMMDD_HHMMSS.csv
```

**Output Analysis**:
- Success rates by method and noise level
- Performance rankings
- Noise robustness metrics
- Method category comparisons

### 4. GP Kernel Comparison (`run_gp_comparison.py`)

**Purpose**: Focused comparison of Gaussian Process kernels to answer specific questions about RBF vs Matérn performance.

**What it does**:
1. Generates synthetic test data (Lotka-Volterra-like function)
2. Adds noise at 7 levels: [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
3. Tests all 7 GP kernel variants
4. Analyzes kernel-specific performance patterns

**Key Insights**:
- RBF vs Matérn performance across noise levels
- Optimal Matérn smoothness parameters
- Noise robustness by kernel type
- Best kernels for different derivative orders

**Usage**:
```bash
python run_gp_comparison.py
# Output: results/gp_comparison_YYYYMMDD_HHMMSS.csv
```

### 5. Unified Comparison (`create_unified_comparison.py`)

**Purpose**: Comprehensive analysis combining ALL methods (Python + Julia + GP variants) into unified comparison.

**What it does**:
1. **Loads all benchmark results**:
   - Python methods from `proper_noisy_benchmark`
   - GP variants from `gp_comparison`
   - Julia methods (estimated from original sweep data)

2. **Creates unified analysis**:
   - Master comparison table with all 20+ methods
   - Implementation comparison (Python vs Julia)
   - Performance by category and derivative order
   - Noise robustness rankings

3. **Generates visualizations**:
   - Performance vs noise level plots
   - Implementation comparison charts
   - Category performance analysis
   - Derivative-specific performance plots

4. **Produces comprehensive report**:
   - Top performers by derivative order
   - Implementation recommendations
   - Key insights and findings

**Usage**:
```bash
python create_unified_comparison.py
# Outputs entire unified_analysis/ directory
```

**Generated Files**:
- `COMPREHENSIVE_REPORT.md`: Main findings and recommendations
- `master_comparison_table.csv`: Complete method comparison
- `unified_comparison_plots.png`: Overview visualizations
- `derivative_performance_plots.png`: Detailed derivative analysis
- `top_performers_derivative_[0-3].csv`: Rankings by derivative order

## Key Algorithms and Approaches

### Method Categories Explained

1. **Interpolation Methods**
   - **CubicSpline**: Piecewise cubic polynomials with C² continuity
   - **SmoothingSpline**: Regularized splines balancing fit vs smoothness
   - **RBF**: Radial basis functions (thin plate spline, multiquadric)

2. **Gaussian Process Methods**
   - **GP_RBF**: Standard GP with RBF kernel
   - **GP_Matérn**: GP with Matérn kernels (different smoothness)
   - **Enhanced variants**: Specialized kernels (periodic, rational quadratic)

3. **Polynomial Methods**
   - **Chebyshev**: Chebyshev polynomial basis expansion
   - **Polynomial**: Standard polynomial fitting with optimal degree

4. **Smoothing Methods**
   - **SavitzkyGolay**: Local polynomial smoothing with analytical derivatives
   - **Butterworth**: Low-pass filtering with finite difference derivatives

5. **Machine Learning**
   - **RandomForest**: Ensemble method with finite difference derivatives
   - **SVR**: Support Vector Regression with kernel tricks

6. **Spectral Methods**
   - **Fourier**: Fourier series with analytical derivative computation

7. **Finite Difference**
   - **FiniteDiff**: Central difference schemes with adaptive step sizes

### Noise Handling Strategy

**Critical Bug Fix**: Early benchmarks incorrectly fitted methods to clean data instead of noisy data, leading to unrealistic perfect performance. The current implementation:

1. **Generates noisy data**: Adds Gaussian noise to clean ODE solutions
2. **Fits to noisy data**: All methods train on the noisy observations
3. **Evaluates against clean truth**: Performance measured against true function/derivatives
4. **Realistic results**: Methods show expected degradation with noise

### Performance Metrics

- **RMSE (Root Mean Square Error)**: Primary metric for ranking methods
- **MAE (Mean Absolute Error)**: Secondary metric for robustness
- **Max Error**: Worst-case performance indicator
- **Timing**: Fit time and evaluation time
- **Success Rate**: Percentage of successful evaluations
- **Robustness Ratio**: Performance degradation with noise (high_noise_RMSE / clean_RMSE)

## Usage Workflows

### 1. Quick Method Comparison
```bash
# Test all Python methods
python run_proper_noisy_benchmark.py

# Check results
ls results/proper_noisy_benchmark_*.csv
```

### 2. GP Kernel Investigation
```bash
# Compare GP kernels
python run_gp_comparison.py

# Analyze GP-specific results
ls results/gp_comparison_*.csv
```

### 3. Comprehensive Analysis
```bash
# Run unified comparison (requires previous benchmarks)
python create_unified_comparison.py

# View complete results
ls unified_analysis/
cat unified_analysis/COMPREHENSIVE_REPORT.md
```

### 4. Custom Method Testing
```python
# Add new method to comprehensive_methods_library.py
class MyNewMethod(DerivativeApproximator):
    def _fit_implementation(self):
        # Your fitting code
        pass
    
    def _evaluate_function(self, t_eval):
        # Your function evaluation
        pass
    
    def _evaluate_derivative(self, t_eval, order):
        # Your derivative evaluation
        pass

# Then re-run benchmarks to include your method
```

## Key Findings Summary

### Performance Rankings (Function Approximation)
1. **AAA_Julia** - Best overall (2.33e-03 RMSE)
2. **GPR_Julia** - Second best (3.88e-03 RMSE)
3. **LOESS_Julia** - Third best (6.20e-03 RMSE)
4. **GP_Periodic** - Best Python method (2.09e-02 RMSE)

### Implementation Insights
- **Julia methods consistently outperform Python** implementations
- **Enhanced GP kernels** provide competitive Python performance
- **Periodic GP kernel** excels for ODE-like data
- **Matérn ν=5.0** best among Matérn variants

### Noise Robustness
- **Smoothing methods** (SavitzkyGolay, Butterworth) most robust
- **Interpolation methods** degrade significantly with noise
- **GP methods** show good noise robustness
- **Finite difference** methods scale directly with noise level

## Dependencies

### Python Environment
```bash
# Create virtual environment
uv venv report-env
source report-env/bin/activate

# Install dependencies
uv pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Required Packages
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization
- `scikit-learn`: Machine learning methods and GP regression
- `scipy`: Scientific computing (interpolation, filtering, optimization)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies installed
2. **Empty results**: Check that Julia benchmark data exists in `results/` directory
3. **Plot generation fails**: Verify matplotlib backend and display settings
4. **Memory issues**: Reduce dataset sizes in benchmark scripts if needed

### Performance Tips

1. **Limit data size**: Use `head()` or sampling for large datasets
2. **Parallel processing**: Methods are independent and can be parallelized
3. **Incremental analysis**: Run benchmarks separately, then combine results
4. **Cache results**: Reuse benchmark outputs for multiple analyses

## Future Extensions

### Potential Improvements
1. **More methods**: Add wavelets, neural networks, physics-informed approaches
2. **Real data**: Test on experimental ODE data beyond synthetic examples
3. **Higher dimensions**: Extend to multivariate time series
4. **Uncertainty quantification**: Include prediction intervals and confidence bounds
5. **Adaptive methods**: Methods that adjust complexity based on noise level

### Research Questions
1. **Optimal noise-dependent method selection**
2. **Hybrid approaches combining multiple methods**
3. **Real-time derivative estimation**
4. **Method performance on different ODE types**

---

This benchmark provides a comprehensive framework for evaluating derivative approximation methods across different implementation languages, noise conditions, and application requirements.