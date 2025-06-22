# DEBUGGING: Derivative Order 4 Investigation

## Issue Description
Only a small subset of methods (8 out of ~24) are attempting to run derivative order 4, while most methods run up to derivative order 3.

## Methods Running Derivative Order 4:
- TVDiff_Julia (72 runs)
- KalmanGrad_Python (72 runs)  
- GPR_Julia (72 runs)
- BSpline5_Julia (72 runs)
- AAA_LS_Python (72 runs)
- AAA_Julia (72 runs)
- AAA_FullOpt_Python (72 runs)
- LOESS_Julia (71 runs)

## Methods NOT Running Derivative Order 4:
All other Python methods including:
- SmoothingSpline, SavitzkyGolay, SVR, RandomForest
- Polynomial, GP_RBF, GP variants, Fourier
- FiniteDiff, CubicSpline, Chebyshev, Butterworth
- RBF variants

## Investigation Findings:
1. Derivative order counts: 0→1871, 1→1727, 2→1727, 3→1727, 4→575
2. The drop from 1727 to 575 at order 4 suggests filtering logic
3. Pattern: Mostly Julia methods + select Python methods (AAA variants, KalmanGrad)

## Key Questions:
1. Is there conditional logic that skips higher derivative orders for certain method types?
2. Are there stability/accuracy thresholds that prevent order 4 evaluation?
3. Is this intentional filtering based on method capabilities?

## Files to Investigate:
- Method evaluation loops in benchmark scripts
- Individual method implementations for derivative order limits
- Error handling that might skip higher orders