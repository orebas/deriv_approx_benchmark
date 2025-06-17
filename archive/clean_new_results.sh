rm -rf results unified_analysis test_data
julia benchmark_derivatives.jl
python3 run_full_benchmark.py
python3 create_unified_comparison.py
