#!/bin/bash

echo "=== BENCHMARK PROGRESS CHECK ==="
echo "$(date)"
echo ""

# Check running processes
echo "1. RUNNING PROCESSES:"
TVDIFF_PID=$(ps aux | grep "run_tvdiff_only" | grep -v grep | awk '{print $2}')
AAA_PID=$(ps aux | grep "run_aaa_smoothbary_only" | grep -v grep | awk '{print $2}')

if [ ! -z "$TVDIFF_PID" ]; then
    echo "   ✓ TVDiff running (PID: $TVDIFF_PID)"
else
    echo "   ✗ TVDiff not running"
fi

if [ ! -z "$AAA_PID" ]; then
    echo "   ✓ AAA_SmoothBary running (PID: $AAA_PID)"
else
    echo "   ✗ AAA_SmoothBary not running"
fi

echo ""

# Check result counts
echo "2. CURRENT RESULTS:"
if [ -f "results/julia_raw_benchmark.csv" ]; then
    JULIA_ROWS=$(wc -l < results/julia_raw_benchmark.csv)
    TVDIFF_COUNT=$(grep -c "TVDiff" results/julia_raw_benchmark.csv 2>/dev/null || echo "0")
    echo "   Julia CSV: $JULIA_ROWS total rows, $TVDIFF_COUNT TVDiff results"
else
    echo "   Julia CSV: not found"
fi

if [ -f "results/python_raw_benchmark.csv" ]; then
    PYTHON_ROWS=$(wc -l < results/python_raw_benchmark.csv)
    AAA_COUNT=$(grep -c "AAA_SmoothBary" results/python_raw_benchmark.csv 2>/dev/null || echo "0")
    echo "   Python CSV: $PYTHON_ROWS total rows, $AAA_COUNT AAA_SmoothBary results"
else
    echo "   Python CSV: not found"
fi

echo ""

# Check log files
echo "3. LOG FILES:"
for log in tvdiff_fixed.log aaa_smoothbary_run.log; do
    if [ -f "$log" ]; then
        SIZE=$(wc -l < "$log")
        echo "   $log: $SIZE lines"
    else
        echo "   $log: not found"
    fi
done

echo ""

# Quick log tail
echo "4. RECENT LOG ACTIVITY:"
if [ -f "tvdiff_fixed.log" ] && [ -s "tvdiff_fixed.log" ]; then
    echo "   TVDiff (last 3 lines):"
    tail -3 tvdiff_fixed.log | sed 's/^/     /'
fi

if [ -f "aaa_smoothbary_run.log" ] && [ -s "aaa_smoothbary_run.log" ]; then
    echo "   AAA_SmoothBary (last 3 lines):"
    tail -3 aaa_smoothbary_run.log | sed 's/^/     /'
fi

echo ""
echo "=== END PROGRESS CHECK ==="