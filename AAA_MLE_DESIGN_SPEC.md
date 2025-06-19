# AAA-MLE Algorithm Design Specification

**Novel Maximum Likelihood Estimation Algorithm for Derivative Approximation**  
*Combining AAA Barycentric Rational Functions with Gaussian Noise Modeling*

---

## Executive Summary

This document specifies a novel algorithm that combines the strengths of Adaptive Anisotropic Approximation (AAA) with Maximum Likelihood Estimation (MLE) and noise-aware model selection. The method uses barycentric rational functions with explicit Gaussian noise modeling to achieve superior derivative approximation for noisy data.

**Key Innovation**: Leverage-based greedy support point addition with noise-aware likelihood optimization.

---

## 1. Mathematical Framework

### 1.1 Model Definition

**Barycentric Rational Function:**
```
r(x; θ) = Σⱼ₌₀ᵐ (wⱼ × fⱼ)/(x - zⱼ) / Σⱼ₌₀ᵐ wⱼ/(x - zⱼ)
```

**Noise Model:**
```
yᵢ = r(xᵢ; θ) + εᵢ,  where εᵢ ~ N(0, σ²)
```

**Parameter Set:**
```
θ = {z₀,...,zₘ, w₀,...,wₘ, f₀,...,fₘ, σ²}
```

### 1.2 Likelihood Function

**Log-Likelihood:**
```
ℓ(θ) = -(N/2)·log(2πσ²) - (1/2σ²)·Σᵢ₌₁ᴺ |yᵢ - r(xᵢ;θ)|²
```

**Parameter Count:**
```
k(m) = 2(m+1) + 2(m+1) + 2(m+1) + 1 = 6(m+1) + 1  [complex case]
k(m) = 3(m+1) + 1                                    [real case]
```

### 1.3 Identifiability Constraints

**Critical Issue (from Gemini)**: Barycentric representation is not unique due to scaling invariance.

**Solution**: Impose normalization constraint
```
w₀ = 1  (fix first weight)
OR
Σⱼ |wⱼ|² = 1  (L2 normalization)
```

---

## 2. Algorithm Overview

### 2.1 Main Algorithm Flow

```
Step 0: AAA_LS Initialization
   - Run existing AAA_LS with small m (1-3 support points)
   - Extract initial θ⁰ = {zⱼ, wⱼ, fⱼ}

Step 1: MLE Refinement at Fixed Support
   - Optimize θ⁰ using alternating optimization (Section 3)
   - Estimate σ² from residuals

Step 2: Greedy Support Addition Loop
   While stopping criteria not met:
     a) Score candidate support points using leverage + residuals
     b) Add 1 or 2 new points (Section 4)
     c) Initialize new weights/values with linear solve
     d) Run local MLE optimization for all parameters
     e) Evaluate AIC/BIC; keep if improved

Step 3: Model Selection & Return
   - Select best model using validation set
   - Return final θ*, σ̂², and uncertainty estimates
```

### 2.2 Stopping Criteria

**Primary**: Validation set negative log-likelihood increases
**Secondary**: Both AIC and BIC increase for 2 consecutive iterations
**Tertiary**: Improvement in -2ℓ̂ < τ (e.g., 1e-3·σ̂²·N)

---

## 3. Optimization Strategy

### 3.1 Alternating Optimization (Recommended)

**Why Alternating** (from o3 + Gemini analysis):
- Leverages problem structure: {wⱼ, fⱼ} subproblem is linear and convex
- Avoids high-dimensional non-convex optimization
- More numerically stable than joint optimization

**Algorithm:**
```
For fixed zⱼ:
  1. Formulate linear system for {wⱼ, fⱼ} products
  2. Solve with Tikhonov regularization: (AᵀA + λI)x = Aᵀb
  3. Extract individual wⱼ, fⱼ with constraint w₀ = 1

For fixed {wⱼ, fⱼ}:
  1. Use L-BFGS-B to optimize {zⱼ, σ²}
  2. Apply domain constraints: zⱼ ∈ [xₘᵢₙ - δ, xₘₐₓ + δ]
  3. Use our smooth_barycentric_eval for gradient computation
```

### 3.2 Numerical Stability Measures

**Critical**: Use our existing smooth barycentric evaluation with W=1e-7
```python
# From our successful implementation
def smooth_barycentric_eval(x, zj, fj, wj, W=1e-7):
    # Already handles near-singularities gracefully
    # Provides smooth gradients for optimization
```

**Additional Safeguards:**
- Regularization parameter: λ = 1e-12 for ill-conditioned linear systems
- Loss clipping: cap individual point loss at 1e6 to prevent NaN gradients
- Domain constraints on support points to prevent poles at infinity

---

## 4. Support Point Addition Strategy

### 4.1 Leverage-Based Scoring

**Score Function:**
```
score(xc) = |residual(xc)| × √(leverage(xc))
```

Where:
- `residual(xc) = (y(xc) - r(xc))/σ̂`
- `leverage(xc)` = diagonal element of hat matrix from current linear solve

**Rationale**: Prioritizes points where model is both wrong AND sensitive

### 4.2 Single vs. Two-Point Addition

**Single Point**: When max residual occurs at interior point
**Two Points**: When top-2 peaks are:
- Symmetric around a potential pole location
- Exhibit steep local curvature indicating sharp feature
- Score ratio > 0.8 and distance < ε_cluster

### 4.3 Duplicate Protection

Skip candidate xc if:
```
min_j |xc - zⱼ| < ε_dist
```
Where ε_dist = 0.01 × (domain_range) / √(current_m)

---

## 5. Model Selection Framework

### 5.1 Information Criteria

**AIC**: `AIC(m) = 2k(m) - 2ℓ̂`
**BIC**: `BIC(m) = k(m)·log(N) - 2ℓ̂`

### 5.2 Validation-Based Selection (Primary)

**Strategy** (from Gemini recommendation):
- Reserve 20% of data as validation set
- Continue greedy addition while validation NLL decreases
- Select model achieving minimum validation NLL
- This prevents overfitting better than AIC/BIC alone

### 5.3 Degrees of Freedom Calculation

**Corrected Parameter Count** (addressing Gemini's concern):
```
k_actual = m + (m-1) + m + 1 = 3m  [real case, w₀=1 constraint]
```
Since fⱼ are not independent (fⱼ = r(zⱼ)), actual DOF ≈ 2m + 1

---

## 6. Implementation Details

### 6.1 JAX Implementation Strategy

**Key Challenges** (from Gemini analysis):
1. NaN gradients from barycentric evaluation → **SOLVED** with our smooth evaluation
2. Dynamic shapes from changing m → Use padding/masking
3. Optimizer state management → Reinitialize BFGS after support point addition

**JAX Patterns:**
```python
# Pre-allocate for maximum expected m
MAX_SUPPORT_POINTS = 50
zj_padded = jnp.zeros(MAX_SUPPORT_POINTS, dtype=jnp.complex128)
active_mask = jnp.zeros(MAX_SUPPORT_POINTS, dtype=bool)

# JIT compile once, reuse with different masks
@jax.jit
def loss_function(params, active_mask):
    zj_active = zj_padded[active_mask]
    # ... rest of computation
```

### 6.2 Robustness Measures

1. **Multi-start**: Run 3-5 random perturbations of initialization
2. **Trust region**: For zⱼ optimization, use trust-region instead of BFGS for better step control
3. **Gradient checking**: Verify finite gradients before each optimization step
4. **Fallback**: If MLE fails, return best AAA_LS result with estimated σ̂²

---

## 7. Theoretical Advantages

### 7.1 vs. Standard AAA
- **Noise-aware**: Explicit σ̂² estimation prevents overfitting noisy regions
- **Principled stopping**: AIC/BIC in likelihood units vs. heuristic tolerance
- **Uncertainty quantification**: Hessian provides standard errors for predictions
- **Adaptive insertion**: Two-point strategy handles steep derivatives better

### 7.2 vs. Gaussian Process Regression
- **Sharp features**: Excellent for discontinuities, poles, high-frequency behavior
- **Compact representation**: Explicit parametric form r(x)
- **Fast evaluation**: O(m) vs. O(N) for GP prediction
- **Analytical derivatives**: Direct computation vs. kernel-dependent smoothness

**Trade-off**: More brittle optimization vs. GP robustness

---

## 8. Expected Challenges & Mitigations

### 8.1 Primary Risks
1. **Non-convex landscape** → Multi-start + good AAA_LS initialization
2. **Pole wandering** → Domain constraints + loss clipping
3. **Overfitting** → Validation set selection + BIC penalty
4. **Numerical instability** → Our smooth evaluation + regularization

### 8.2 Fallback Strategies
- If optimization fails: return AAA_LS result
- If evaluation becomes unstable: reduce W parameter
- If overfitting detected: revert to simpler model

---

## 9. Implementation Roadmap

### Phase 1: Core Framework (Week 1-2)
1. Implement alternating optimizer with our smooth_barycentric_eval
2. Add weight constraint (w₀ = 1)
3. Verify on synthetic data with known σ

### Phase 2: Greedy Addition (Week 3)
1. Implement leverage-based scoring
2. Add single-point insertion logic
3. Test on polynomial and rational test functions

### Phase 3: Model Selection (Week 4)
1. Add AIC/BIC computation with correct DOF
2. Implement validation set approach
3. Comprehensive testing vs. AAA_LS baseline

### Phase 4: Advanced Features (Week 5+)
1. Two-point insertion heuristics
2. Uncertainty quantification
3. Integration with benchmark framework

---

## 10. Success Metrics

**Technical**:
- Convergence rate > 90% on synthetic test suite
- RMSE improvement > 10% vs. AAA_LS on noisy data
- No NaN/Inf outputs in derivative computation

**Practical**:
- Runtime < 5× AAA_LS for equivalent accuracy
- Stable behavior across noise levels σ ∈ [1e-4, 1e-1]
- Integration with existing benchmark framework

---

## Conclusion

This AAA-MLE algorithm represents a significant theoretical advance, combining the flexibility of rational approximation with rigorous statistical modeling. The design addresses key challenges through:

1. **Our proven smooth barycentric evaluation** (solving the biggest implementation hurdle)
2. **Alternating optimization** (leveraging problem structure)
3. **Validation-based model selection** (preventing overfitting)
4. **Leverage-aware point addition** (efficient support point placement)

**Next Steps**: Begin Phase 1 implementation with focus on robust alternating optimization and validation against known test cases.

---

*Design review completed by o3 (mathematical framework) and Gemini Pro (critical analysis)*  
*Implementation ready for development team*