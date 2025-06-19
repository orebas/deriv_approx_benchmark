#!/bin/bash

echo "Running missing methods benchmark..."
echo "This will take a while. Running in background..."

# Run TVDiff benchmark
echo "Starting TVDiff benchmark..."
nohup julia run_tvdiff_only.jl > tvdiff_run.log 2>&1 &
TVDIFF_PID=$!
echo "TVDiff running with PID: $TVDIFF_PID"

# Run AAA_SmoothBary benchmark (in virtual env)
echo "Starting AAA_SmoothBary benchmark..."
source report-env/bin/activate
nohup python3 run_aaa_smoothbary_only.py > aaa_smoothbary_run.log 2>&1 &
AAA_PID=$!
echo "AAA_SmoothBary running with PID: $AAA_PID"

echo ""
echo "Both benchmarks are running in the background."
echo "Monitor progress with:"
echo "  tail -f tvdiff_run.log"
echo "  tail -f aaa_smoothbary_run.log"
echo ""
echo "Check if still running with:"
echo "  ps -p $TVDIFF_PID,$AAA_PID"
echo ""
echo "When complete, run:"
echo "  python3 create_unified_comparison.py"