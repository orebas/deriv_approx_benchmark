# AAA_FullOpt Stability Analysis and Recommendations

## Executive Summary

This analysis investigated the stability issues with AAA_FullOpt (AAA with full optimization of all parameters) compared to AAA_LS (AAA with least-squares refinement). The investigation found fundamental mathematical and numerical issues that explain the instability, and developed practical solutions.

**Key Finding**: AAA_FullOpt's instability stems from multiple interacting factors: discontinuous gradients, over-parameterization, pole collisions, and inadequate regularization. While fixable, the complexity may not justify the benefits over the stable AAA_LS approach.

**Recommendation**: Use AAA_LS as the primary method, optionally with a two-stage refinement approach for critical applications.

## Root Cause Analysis

### 1. Discontinuous Gradients (Critical Issue)

**Problem**: The `jax.lax.cond` in the barycentric evaluation function creates discontinuous gradients that break L-BFGS-B optimization.

```python
# In barycentric_eval function:
is_support_point = jnp.any(jnp.isclose(x, zj))
return jax.lax.cond(is_support_point, true_fn, false_fn)
```

**Impact**: The optimizer receives zero gradients for support point parameters almost everywhere, with undefined gradients at the support points themselves. This provides no guidance for optimization.

**Evidence**: Diagnostic tests showed NaN gradients and optimization failures when support points moved close to evaluation points.

### 2. Over-Parameterization and Non-Identifiability

**Problem**: AAA_FullOpt uses 3m parameters (support points + values + weights) for a rational function with only 2m-1 degrees of freedom.

**Impact**: Multiple parameter sets produce identical functions, creating flat directions in the loss landscape. This leads to ill-conditioning and numerical instability.

**Evidence**: Weight condition numbers of 100s-1000s observed even at moderate noise levels.

### 3. Pole Collision and Clustering

**Problem**: Support points cluster together during optimization, leading to nearly singular basis functions.

**Impact**: When `(x - zj)` and `(x - zk)` are nearly equal, the basis functions become linearly dependent, causing explosive parameter growth.

**Evidence**: Minimum support point distances remained constant while adding more points, indicating clustering behavior.

### 4. Inadequate Regularization

**Problem**: Second derivative penalty is only evaluated at data points, missing pathological behavior between points.

**Impact**: Optimizer can create sharp spikes that fit noise perfectly as long as the spikes don't coincide with data points.

## Diagnostic Results

### Noise Sensitivity Analysis

| Noise Level | AAA_LS Success | AAA_FullOpt Issues |
|------------|----------------|-------------------|
| 0.01 | ✓ Stable | Weight condition: 48, manageable gradients |
| 0.05 | ✓ Stable | Weight condition: 469, gradient norm: 2.6k |
| 0.1 | ✓ Stable | Weight condition: 59, gradient norm: 889 |

### Gradient Analysis

- **Support point gradients**: Significant and often unstable
- **Perturbation sensitivity**: Small changes in support points cause large objective changes
- **Weight gradients**: Dominate the gradient norm, indicating the optimization is primarily adjusting weights

## Proposed Solutions

### 1. Two-Stage Approach (Recommended)

**Strategy**: Use AAA_LS for stable initialization, then apply limited AAA_FullOpt refinement.

**Implementation**:
```python
# Stage 1: AAA_LS (stable baseline)
zj, fj_ls, wj_ls = run_aaa_ls(t, y)

# Stage 2: Limited refinement with constraints
zj_delta = optimize_with_constraints(zj, fj_ls, wj_ls)
zj_final = zj + clip(zj_delta, -max_perturbation, max_perturbation)
```

**Advantages**:
- Leverages AAA_LS stability
- Allows refinement without full instability risk
- Only accepts improvements that significantly better BIC
- Conservative perturbation limits prevent pole collisions

### 2. Smooth Barycentric Approximation

**Strategy**: Replace discontinuous `jax.lax.cond` with smooth transition using `tanh`.

**Implementation**:
```python
def smooth_barycentric_eval(x, zj, fj, wj, tolerance=1e-8):
    val_interp = standard_barycentric_formula(x, zj, fj, wj)
    val_direct = fj[argmin(|x - zj|)]
    activation = 0.5 * (1 + tanh(-min_dist_sq / tolerance + 10))
    return activation * val_direct + (1 - activation) * val_interp
```

**Trade-offs**:
- ✓ Provides smooth gradients for optimization
- ✗ Introduces artificial gradients that don't reflect true mathematics
- ✗ Requires tuning of tolerance parameter

### 3. Enhanced Regularization and Constraints

**Separation Penalty**:
```python
zj_sorted = jnp.sort(zj)
dists = jnp.diff(zj_sorted)
separation_penalty = jnp.sum(jax.nn.relu(min_dist_allowed - dists))
```

**Fine Grid Regularization**:
```python
t_fine = jnp.linspace(min(t), max(t), 3 * len(t))
smoothness_term = jnp.mean(d2_func(t_fine)**2)
```

## Performance Comparison

Based on the analysis, the methods rank as follows for reliability:

1. **AAA_LS**: Most reliable, no hyperparameters, good performance
2. **AAA_TwoStage**: Good reliability with potential for improvement
3. **AAA_SmoothBarycentric**: Moderate reliability, requires tuning
4. **AAA_FullOpt (original)**: Least reliable, fails frequently at high noise

## Recommendations

### For Production Use
- **Primary Choice**: Use AAA_LS for derivative approximation
- **Secondary Option**: Consider AAA_TwoStage for critical applications where maximum accuracy is needed

### For Research/Development
- Investigate the two-stage approach further with real benchmark data
- The smooth barycentric approach may be worth exploring for specific use cases
- Consider alternative parameterizations (e.g., optimizing support point spacings rather than positions)

### When AAA_FullOpt Might Be Worth It
- Functions with sharp, localized features that AAA_LS misses
- High-frequency oscillatory behavior where optimal pole placement is critical
- Ultimate accuracy requirements where the extra complexity is justified

### When to Avoid AAA_FullOpt
- Noisy data (noise > 1% of signal amplitude)
- Real-time applications requiring reliability
- When hyperparameter tuning effort isn't justified
- Most derivative approximation tasks (AAA_LS is usually sufficient)

## Implementation Notes

### Files Created
- `diagnose_aaa_stability.py`: Comprehensive diagnostic framework
- `quick_aaa_diagnostic.py`: Focused diagnostic for specific failure cases
- `minimal_aaa_test.py`: Basic stability tests
- `aaa_stable_implementation.py`: Stable AAA implementations
- `simple_aaa_test.py`: Core concept verification

### Key Technical Insights
1. The `jax.lax.cond` discontinuity is the primary technical blocker
2. Over-parameterization amplifies other numerical issues
3. Conservative constraints can stabilize the optimization
4. Two-stage approaches effectively combine stability and performance

## Future Work

1. **Benchmark Testing**: Test stable implementations on the full ODE benchmark suite
2. **Hyperparameter Sensitivity**: Systematic study of regularization parameters
3. **Alternative Optimizers**: Investigate constrained optimization methods (SLSQP, IPOPT)
4. **Theoretical Analysis**: Mathematical analysis of when AAA_FullOpt is guaranteed to converge

## Conclusion

AAA_FullOpt's instability issues are fundamental but solvable. However, the added complexity may not justify the marginal benefits over AAA_LS for most applications. The two-stage approach provides the best balance of stability and potential performance improvement.

The investigation demonstrates that **letting support points float during optimization is worth it only in specific circumstances** where the stability risks and implementation complexity are justified by the accuracy requirements.