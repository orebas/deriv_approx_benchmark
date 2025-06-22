# Scripts Guide: Derivative Approximation Benchmark

Quick reference for understanding and using each script in the benchmark suite.

## 📚 Core Libraries

### `comprehensive_methods_library.py`
**What it is**: The heart of the Python implementation - contains all 14 derivative approximation methods.

**Key Classes**:
- `DerivativeApproximator`: Base class that all methods inherit from
- `CubicSplineApproximator`, `GPApproximator`, etc.: Individual method implementations

**Main Functions**:
- `create_all_methods(t, y)`: Creates instances of all 14 methods for given data
- `get_method_categories()`: Returns method groupings by category

**How methods work**:
1. **Initialization**: `method = CubicSplineApproximator(t, y, "CubicSpline")`
2. **Fitting**: Happens automatically in `__init__`, calls `_fit_implementation()`
3. **Evaluation**: `results = method.evaluate(t_eval, max_derivative=3)`
4. **Results**: Dictionary with `{'y': func_values, 'd1': first_deriv, 'd2': second_deriv, 'd3': third_deriv, 'success': bool}`

**Adding new methods**:
```python
class MyMethod(DerivativeApproximator):
    def _fit_implementation(self):
        # Your fitting logic here
        self.fitted_model = fit_my_model(self.t, self.y)
    
    def _evaluate_function(self, t_eval):
        # Return function values at t_eval
        return self.fitted_model.predict(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        # Return derivative values at t_eval
        return compute_derivative(self.fitted_model, t_eval, order)
```

### `enhanced_gp_methods.py`
**What it is**: Specialized Gaussian Process implementations for detailed kernel comparison.

**Key Class**: `EnhancedGPApproximator` - GP with configurable kernels

**Kernel Types**:
- `'rbf_iso'`: Isotropic RBF kernel
- `'matern_0.5'`, `'matern_1.5'`, `'matern_2.5'`, `'matern_5.0'`: Matérn with different smoothness
- `'rational_quadratic'`: Mixture of RBF length scales
- `'periodic'`: For periodic functions

**Usage**:
```python
# Create single GP method
gp = EnhancedGPApproximator(t, y, "MyGP", kernel_type='rbf_iso')

# Create all GP variants
methods = create_enhanced_gp_methods(t, y)
```

## 🧪 Benchmark Scripts

### `run_proper_noisy_benchmark.py`
**Purpose**: Main Python benchmark testing all methods on realistic noisy data.

**Workflow**:
1. **Data Generation**: `generate_noisy_test_cases()`
   - Loads clean ODE data from Julia results
   - Adds Gaussian noise: `noise = np.random.normal(0, noise_std, len(y))`
   - Creates test cases for noise levels: [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2]

2. **Benchmarking**: `run_proper_noisy_benchmark()`
   - For each noise level and method:
     - Fits method to **noisy data**: `methods = create_all_methods(t, y_noisy)`
     - Evaluates against **clean truth**: `errors = y_pred - y_true`
     - Computes RMSE, MAE, max error
   - Tests function + 1st/2nd/3rd derivatives

3. **Analysis**: `analyze_proper_results()`
   - Performance rankings by noise level
   - Robustness analysis (high_noise_RMSE / clean_RMSE)
   - Top performers by method category

**Critical Fix**: Early versions incorrectly used `y = subset['true_value'].values` (clean data). Fixed to use proper noisy data.

**Command**: `python run_proper_noisy_benchmark.py`
**Output**: `results/proper_noisy_benchmark_YYYYMMDD_HHMMSS.csv`

### `run_gp_comparison.py`
**Purpose**: Focused comparison of GP kernels to answer "RBF vs Matérn" questions.

**Workflow**:
1. **Test Data**: `generate_test_data()`
   - Creates synthetic Lotka-Volterra-like function
   - `y = 2 + 3*sin(t) + 1.5*cos(2*t) + 0.5*sin(3*t)`
   - Adds noise at 7 levels: [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

2. **GP Testing**: `run_gp_comparison()`
   - Tests all 7 GP kernel variants
   - Evaluates function + 3 derivative orders
   - Measures performance and timing

3. **Analysis**: `analyze_gp_results()`
   - Performance by noise level
   - Robustness ranking (degradation ratios)
   - Best kernel by derivative order
   - Matérn smoothness parameter comparison

**Key Insights Generated**:
- Which GP kernel performs best at each noise level
- Matérn smoothness parameter effects (ν = 0.5, 1.5, 2.5, 5.0)
- Noise robustness by kernel type

**Command**: `python run_gp_comparison.py`
**Output**: `results/gp_comparison_YYYYMMDD_HHMMSS.csv`

### `create_unified_comparison.py`
**Purpose**: Master analysis combining ALL methods (Python + Julia + GP) into comprehensive comparison.

**Workflow**:
1. **Data Loading**: `load_all_results()`
   - Python methods from `proper_noisy_benchmark_*.csv`
   - GP variants from `gp_comparison_*.csv`
   - Julia methods (estimated from original `sweep_lv_periodic_*.csv`)

2. **Unified Analysis**: `create_unified_analysis()`
   - Master comparison table with all ~23 methods
   - Implementation comparison (Python vs Julia for same algorithms)
   - Performance by category and derivative order
   - Robustness rankings

3. **Visualizations**: Creates 4 main plots:
   - Performance vs noise level (top methods)
   - Python vs Julia implementation comparison
   - Performance by method category
   - Robustness analysis (degradation ratios)

4. **Report Generation**: `create_final_report()`
   - Comprehensive markdown report
   - Top performers by derivative order
   - Implementation recommendations
   - Key insights and findings

**Command**: `python create_unified_comparison.py`
**Output**: Entire `unified_analysis/` directory with plots, tables, and report

## 📊 Understanding the Output Files

### CSV Result Files
All benchmark results follow this structure:
- `method`: Method name (e.g., "CubicSpline", "GP_RBF_Iso")
- `implementation`: "Python", "Julia", or "Python_GP"
- `category`: Method category (e.g., "Interpolation", "Gaussian_Process")
- `noise_level`: Noise level tested (0.0 to 0.1)
- `derivative_order`: 0 (function), 1, 2, or 3
- `rmse`: Root mean square error (primary metric)
- `mae`: Mean absolute error
- `max_error`: Maximum absolute error
- `eval_time`: Evaluation time in seconds
- `fit_time`: Fitting time in seconds
- `success`: Boolean indicating if method succeeded

### Analysis Files
- `master_comparison_table.csv`: Complete method comparison with averages
- `implementation_comparison.csv`: Direct Python vs Julia comparisons
- `top_performers_derivative_[0-3].csv`: Best methods for each derivative order

### Visualization Files
- `unified_comparison_plots.png`: 4-panel overview (performance, implementation, category, robustness)
- `derivative_performance_plots.png`: 4-panel derivative-specific analysis

## 🛠️ Customization and Extension

### Adding New Methods
1. **Create method class** in `comprehensive_methods_library.py`:
```python
class MyNewMethod(DerivativeApproximator):
    def _fit_implementation(self):
        # Fit your model to self.t, self.y
        pass
    
    def _evaluate_function(self, t_eval):
        # Return function predictions
        pass
    
    def _evaluate_derivative(self, t_eval, order):
        # Return derivative predictions
        pass
```

2. **Add to method creation**:
```python
def create_all_methods(t, y):
    methods = {}
    # ... existing methods ...
    methods['MyNewMethod'] = MyNewMethod(t, y, "MyNewMethod")
    return methods
```

3. **Update categories**:
```python
def get_method_categories():
    return {
        # ... existing categories ...
        'My_Category': ['MyNewMethod']
    }
```

### Modifying Benchmarks
- **Change noise levels**: Edit `noise_levels` arrays in benchmark scripts
- **Add test functions**: Modify data generation functions
- **Change evaluation points**: Edit `t_eval` arrays
- **Add metrics**: Extend result dictionaries with new performance measures

### Custom Analysis
```python
# Load existing results
df = pd.read_csv('results/proper_noisy_benchmark_latest.csv')

# Filter for specific analysis
func_data = df[df['derivative_order'] == 0]
low_noise = func_data[func_data['noise_level'] <= 0.01]

# Custom analysis
best_methods = low_noise.groupby('method')['rmse'].mean().sort_values()
print(best_methods.head())
```

## 🔧 Troubleshooting

### Common Errors and Solutions

1. **"ModuleNotFoundError: No module named 'sklearn'"**
   ```bash
   source report-env/bin/activate
   uv pip install scikit-learn
   ```

2. **"FileNotFoundError: No such file or directory: 'results/...'"**
   - Run the prerequisite benchmarks first
   - Check that Julia data exists for unified comparison

3. **"Memory Error" or slow performance**
   - Reduce dataset sizes in benchmark scripts
   - Limit number of methods tested
   - Use sampling for large datasets

4. **Plot generation fails**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Non-interactive backend
   ```

5. **RBF_Multiquadric errors about epsilon**
   - This is expected - the method needs epsilon parameter
   - Error is handled gracefully in results

### Performance Optimization
- **Parallel processing**: Methods are independent and can be run in parallel
- **Incremental analysis**: Run benchmarks separately, combine results later
- **Caching**: Reuse fitted models where possible
- **Sampling**: Use subset of data for quick tests

---

This guide provides the technical details needed to understand, modify, and extend the derivative approximation benchmark suite.