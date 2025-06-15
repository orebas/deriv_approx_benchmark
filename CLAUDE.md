# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a comprehensive benchmarking framework for derivative approximation methods, with implementations in both Julia and Python. The project evaluates various methods for approximating derivatives of noisy time series data, particularly from ODE systems.

## Common Development Tasks

### Running Tests

**Julia tests:**
```bash
julia test_run.jl  # Simple test with minimal configuration
julia test_aaa_derivatives.jl  # Test AAA method derivatives
```

**Python benchmarks:**
```bash
# Activate virtual environment first
source report-env/bin/activate

# Run main benchmarks
python run_proper_noisy_benchmark.py  # Test all Python methods with noise
python run_gp_comparison.py  # Compare GP kernel variants
python create_unified_comparison.py  # Generate comprehensive analysis
```

### Building and Running Full Benchmark

To run the complete benchmark suite:
```bash
# Clean previous results and run full benchmark
./clean_new_results.sh
```

This script:
1. Removes old results
2. Runs Julia benchmark (`julia benchmark_derivatives.jl`)
3. Runs Python benchmark (`python3 run_full_benchmark.py`)
4. Creates unified comparison (`python3 create_unified_comparison.py`)

### Linting and Type Checking

**Python:**
```bash
# No specific linting command found - use standard Python tools if needed
python -m pylint *.py  # If pylint is installed
python -m mypy *.py    # If mypy is installed
```

**Julia:**
```bash
# Julia doesn't have standard linting commands
# Ensure code follows Julia style guide manually
```

## High-Level Architecture

### Core Components

1. **Julia Implementation** (`src/` directory):
   - `DerivativeApproximationBenchmark.jl`: Main module providing benchmark framework
   - `approximation_methods.jl`: Julia methods (AAA, GPR, LOESS, BSpline, etc.)
   - `data_generation.jl`: Synthetic ODE data generation with noise
   - `evaluation.jl`: Error metrics and evaluation framework
   - `builtin_examples.jl`: Pre-defined ODE systems (Lotka-Volterra, SIR, etc.)

2. **Python Implementation**:
   - `comprehensive_methods_library.py`: 14 Python methods with unified `DerivativeApproximator` interface
   - `enhanced_gp_methods.py`: Specialized GP implementations with various kernels
   - `python_methods_bridge.jl`: Bridge for calling Python from Julia

3. **Benchmark Orchestration**:
   - Methods are tested on noisy data generated from known ODE solutions
   - Performance evaluated by fitting to noisy data, comparing against clean truth
   - Results stored in CSV format for analysis

### Method Categories

- **Interpolation**: CubicSpline, SmoothingSpline, RBF variants
- **Gaussian Process**: Different kernels (RBF, Matérn, Periodic)
- **Polynomial**: Chebyshev, standard polynomial fitting
- **Smoothing**: Savitzky-Golay, Butterworth filter
- **Machine Learning**: Random Forest, SVR
- **Spectral**: Fourier series
- **Finite Difference**: Central difference schemes

### Data Flow

1. **ODE System** → Clean solution via numerical integration
2. **Add Noise** → Create realistic test data at various noise levels
3. **Fit Methods** → Each method fits to noisy data
4. **Evaluate** → Compare predictions against clean truth
5. **Analyze** → Aggregate results, generate reports and visualizations

### Key Design Decisions

- **Separate Implementations**: Julia and Python methods kept separate for fair comparison
- **Unified Interface**: All methods follow same evaluation protocol
- **Realistic Testing**: Methods fit to noisy data, evaluated against clean truth
- **Comprehensive Metrics**: RMSE, MAE, max error, timing, success rates
- **Modular Design**: Easy to add new methods or test cases

## Important Notes

- Virtual environment `report-env/` contains Python dependencies
- Results stored in `results/` directory as timestamped CSV files
- `unified_analysis/` contains final comprehensive reports and visualizations
- Test data in `test_data/` includes pre-generated ODE solutions at various noise levels