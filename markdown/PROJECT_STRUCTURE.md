# Project Structure Overview

## 🚀 **Core Production System**

### **Main Entry Points**
```
./run_unified_benchmark.py     # Main benchmark runner (USE THIS!)
./benchmark_derivatives.jl     # Julia benchmark component  
./run_full_benchmark.py        # Python benchmark component
./create_unified_comparison.py # Results analysis
```

### **Configuration**
```
benchmark_config.json          # Main configuration file
CLAUDE.md                      # Developer instructions
```

### **Method Libraries**
```
comprehensive_methods_library.py  # All Python methods
enhanced_gp_methods.py            # Gaussian Process variants
src/julia_aaa_final.jl            # Julia AAA implementations ⭐
```

## 📁 **Source Code (src/)**

```
src/
├── julia_aaa_final.jl           # 🎯 YOUR JULIA AAA METHODS
├── approximation_methods.jl     # Method dispatcher & evaluation  
├── builtin_examples.jl          # ODE problem definitions
├── data_generation.jl           # Test data generation
├── evaluation.jl                # Error calculation & metrics
└── examples/models/             # ODE model implementations
    ├── classical_systems.jl     # Basic ODE systems
    ├── biological_systems.jl    # Bio models
    └── advanced_systems.jl      # Complex systems
```

## 🗄️ **Generated Data & Results**

```
results/                        # Raw benchmark results
test_data/                      # Generated test datasets  
unified_analysis/               # Analysis reports & logs
```

## 📚 **Archive (Moved from Root)**

All development/debug files have been organized in `archive/`:

```
archive/
├── debug_scripts/              # All debug_*.py, fix_*.py files
├── test_scripts/               # All test_*.py files  
├── utilities/                  # Helper scripts & tools
├── obsolete_implementations/   # Old AAA files & experiments
├── old_configs/               # Backup config files
└── [previous reports]/        # Historical analysis reports
```

## 🎯 **Quick Start**

1. **Run benchmark**: `./run_unified_benchmark.py`
2. **Check results**: Look in `unified_analysis/` for reports
3. **Modify methods**: Edit `src/julia_aaa_final.jl` for Julia AAA methods
4. **Add Python methods**: Edit `comprehensive_methods_library.py`

## ⚠️ **DO NOT MODIFY**
- `src/approximation_methods.jl` (method integration)
- `benchmark_derivatives.jl` (Julia runner)  
- The main benchmark scripts (working system)

## 🧹 **Cleanup Summary**

**Moved to Archive:**
- ✅ 15+ debug scripts (`debug_*.py`)
- ✅ 25+ test scripts (`test_*.py`) 
- ✅ 2 obsolete AAA implementations (`aaa_methods_julia.jl`, `simple_aaa_julia.jl`)
- ✅ Utility scripts and old configs
- ✅ Development experiments and fixes

**Result:** Clean, focused project structure with clear separation between production code and development artifacts.