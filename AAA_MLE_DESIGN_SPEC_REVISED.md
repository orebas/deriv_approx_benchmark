# AAA-MLE Algorithm Design Specification (REVISED)

**Practical Maximum Likelihood Estimation for Derivative Approximation**  
*Simplified Design Based on Proven AAA Methods + MLE Refinement*

---

## Executive Summary

**MAJOR REVISION**: Based on consensus from o3, Gemini, and user feedback, this specification dramatically simplifies the original design while retaining the core innovation of MLE refinement.

**Key Changes**:
- ✅ **Start with full AAA_LS result** (~10-20 points) instead of growing from m=1-3
- ✅ **Use simple residual-based point addition** (`argmax |residual|`) instead of leverage scoring
- ✅ **Keep MLE optimization and noise modeling** for superior fitting
- ✅ **Maintain alternating optimization** for numerical stability

**Result**: Much faster, simpler, more maintainable algorithm with proven AAA foundation.

---

## 1. Simplified Mathematical Framework

### 1.1 Model Definition (Unchanged)

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

### 1.2 Identifiability Constraint
```
w₀ = 1  (fix first weight for uniqueness)
```

---

## 2. Simplified Algorithm Overview

### 2.1 Main Algorithm Flow (REVISED)

```
Step 0: Full AAA_LS Initialization (USER SUGGESTION)
   - Run complete AAA_LS to convergence (~10-20 support points)
   - Extract θ⁰ = {zⱼ, wⱼ, fⱼ} from AAA_LS result
   - Time savings: ~10x faster than growing from m=1-3

Step 1: MLE Refinement 
   - Optimize θ⁰ using alternating optimization
   - Estimate σ² from residuals
   - Apply w₀ = 1 constraint

Step 2: Simple Greedy Addition (PROVEN AAA METHOD)
   While validation loss decreases:
     a) Find max residual: new_point = argmax |yᵢ - r(xᵢ)|
     b) Add single support point at new_point  
     c) Re-initialize weights with linear solve
     d) Run local MLE optimization
     e) Check validation set for improvement

Step 3: Return Result
   - Final θ*, σ̂², and uncertainty estimates
```

### 2.2 Why This is Much Better

**Performance** (from o3 analysis):
- AAA_LS: ~30ms for full initialization
- MLE iteration: 50-500ms each
- **Time savings**: Start with good 20-point basis vs. 15+ slow iterations from m=1

**Simplicity** (consensus view):
- Proven AAA point selection vs. complex leverage computation
- Standard `argmax |residual|` vs. `|residual| × √(leverage)`
- **Maintenance**: Easier to debug, explain, and extend

**Quality** (o3 benchmarking):
- Leverage scoring: <4% improvement in 2/7 cases, worse in 1/7
- Runtime cost: +18% for marginal gains
- **Conclusion**: Not worth the complexity

---

## 3. Optimization Strategy (Simplified)

### 3.1 Alternating Optimization (Kept)

**Still the best approach**:
```
For fixed zⱼ:
  1. Solve linear system: (AᵀA + λI)x = Aᵀb for {wⱼ, fⱼ}
  2. Apply constraint: w₀ = 1

For fixed {wⱼ, fⱼ}:
  1. L-BFGS-B on {zⱼ, σ²} 
  2. Use smooth_barycentric_eval(W=1e-7) for gradients
```

### 3.2 Numerical Stability (Proven Methods)

**Use our existing infrastructure**:
```python
# Already tested and working
from comprehensive_methods_library import smooth_barycentric_eval

# Stable evaluation with smooth gradients
def evaluate_model(x, zj, fj, wj):
    return smooth_barycentric_eval(x, zj, fj, wj, W=1e-7)
```

---

## 4. Point Addition Strategy (VASTLY SIMPLIFIED)

### 4.1 Proven AAA Method

**Standard Approach** (user insight):
```python
def find_next_support_point(x_data, y_data, current_model):
    """Use proven AAA strategy - simple and effective"""
    residuals = abs(y_data - current_model(x_data))
    return x_data[argmax(residuals)]
```

**Why This Works** (user observation):
- Picks point "farthest from current approximant"
- Has worked "surprisingly well" in AAA for decades
- Simple, fast, maintainable

### 4.2 Leverage Scoring (REMOVED)

**Decision**: Removed based on consensus
- **o3 evidence**: <4% improvement, +18% cost
- **Gemini assessment**: "Burden of proof on complex method"
- **User wisdom**: "AAA works surprisingly well"

---

## 5. Model Selection (Streamlined)

### 5.1 Validation-Based Primary Selection

**Simple and Effective**:
```python
# Reserve 20% for validation
val_indices = random.sample(range(N), N//5)
train_indices = remaining_indices

# Continue while validation improves
while val_loss_decreases:
    add_support_point()
    reoptimize()
    
# Select model with minimum validation loss
best_model = min(models, key=lambda m: m.val_loss)
```

### 5.2 Information Criteria (Secondary)

**Backup stopping criteria**:
- AIC/BIC with corrected DOF count
- Stop if both increase for 2 consecutive iterations

---

## 6. Implementation Roadmap (UPDATED)

### Phase 1: Core Framework (Week 1)
```python
def AAA_MLE_v1(t, y):
    # 1. Full AAA_LS initialization (USER SUGGESTION)
    aaa_ls = AAALeastSquaresApproximator(t, y)
    aaa_ls.fit()
    
    # 2. MLE refinement with alternating optimization
    theta = mle_refine(aaa_ls.zj, aaa_ls.fj, aaa_ls.wj)
    
    return theta
```

### Phase 2: Greedy Addition (Week 2)
```python
def AAA_MLE_v2(t, y):
    theta = AAA_MLE_v1(t, y)  # Start with refined AAA_LS
    
    # 3. Simple greedy addition (PROVEN AAA METHOD)
    while validation_improves:
        new_point = argmax_residual(t, y, theta)  # Simple!
        theta = add_point_and_reoptimize(theta, new_point)
        
    return theta
```

### Phase 3: Integration (Week 3)
- Add to comprehensive_methods_library.py
- Integrate with benchmark framework
- Comprehensive testing vs. AAA_LS baseline

---

## 7. Expected Performance

### 7.1 Speed Improvements

**Initialization speedup** (user insight):
- Old: 15+ iterations × 50-500ms = **7.5-75 seconds**
- New: AAA_LS (30ms) + 3-5 refinements = **150-2500ms**
- **Improvement**: 5-30x faster initialization

**Point addition speedup** (consensus):
- Old: Leverage computation O(nm²) per step
- New: Simple residual argmax O(n) per step  
- **Improvement**: 10-20x faster per addition step

### 7.2 Quality Expectations

**Should match or exceed AAA_LS** because:
- Same intelligent support point placement
- **Plus** explicit noise modeling (σ² estimation)
- **Plus** MLE optimization of all parameters
- **Plus** principled validation-based stopping

---

## 8. Code Structure

### 8.1 Class Definition

```python
class AAA_MLE_Approximator(DerivativeApproximator):
    """
    Simplified AAA-MLE using proven AAA methods + MLE refinement
    
    Key simplifications:
    - Start with full AAA_LS result (user suggestion)
    - Use simple argmax(residual) point addition (consensus)
    - Keep alternating optimization for stability
    """
    
    def __init__(self, t, y, name="AAA_MLE"):
        super().__init__(t, y, name)
        self.max_derivative_supported = 7
        
    def _fit_implementation(self):
        # Step 1: Full AAA_LS initialization
        aaa_ls = AAALeastSquaresApproximator(self.t, self.y)
        aaa_ls.fit()
        
        # Step 2: MLE refinement  
        self.zj, self.fj, self.wj, self.sigma = self._mle_refine(
            aaa_ls.zj, aaa_ls.fj, aaa_ls.wj
        )
        
        # Step 3: Greedy addition with validation
        self._greedy_addition_simple()
        
        # Step 4: Build derivative functions
        self._build_derivatives()
```

### 8.2 Core Methods

```python
def _mle_refine(self, zj_init, fj_init, wj_init):
    """Alternating MLE optimization"""
    # Implementation using our smooth_barycentric_eval
    
def _find_next_point(self):
    """Simple AAA-style point selection"""
    residuals = abs(self.y - self._evaluate_batch(self.t))
    return self.t[jnp.argmax(residuals)]
    
def _evaluate_batch(self, t_eval):
    """Vectorized evaluation using smooth method"""
    return jax.vmap(lambda x: smooth_barycentric_eval(
        x, self.zj, self.fj, self.wj, W=1e-7))(t_eval)
```

---

## 9. Success Metrics (Realistic)

**Performance**:
- ✅ **Initialization**: 5-30x faster than growing from m=1-3
- ✅ **Point addition**: 10-20x faster than leverage-based
- ✅ **Overall runtime**: <2x AAA_LS for equivalent accuracy

**Quality**:
- ✅ **RMSE**: Match or exceed AAA_LS (due to noise modeling)
- ✅ **Derivatives**: All finite using smooth evaluation
- ✅ **Robustness**: No convergence failures

**Simplicity**:
- ✅ **Code complexity**: Similar to AAA_LS  
- ✅ **Maintainability**: Standard patterns, no complex heuristics
- ✅ **Debuggability**: Clear algorithm steps

---

## 10. Theoretical Advantages (Preserved)

### 10.1 vs. Standard AAA
- ✅ **Noise modeling**: Explicit σ̂² estimation  
- ✅ **Parameter optimization**: All θ = {zⱼ, wⱼ, fⱼ} refined via MLE
- ✅ **Validation stopping**: Data-driven model selection
- ✅ **Uncertainty**: Hessian-based standard errors

### 10.2 vs. Original Complex Design  
- ✅ **Speed**: 5-30x faster initialization
- ✅ **Simplicity**: Standard AAA patterns vs. leverage complexity  
- ✅ **Reliability**: Proven methods vs. experimental heuristics
- ✅ **Maintainability**: Clear code vs. complex scoring functions

---

## Conclusion

**This revised design represents the best of both worlds:**

1. **User's practical engineering insight**: Use proven AAA initialization and point selection
2. **Theoretical advancement**: Add MLE framework for noise modeling and parameter optimization  
3. **Consensus simplification**: Remove complex heuristics that don't provide clear value

**Key Decision**: Start with what works (AAA_LS + simple greedy) and add value (MLE + noise modeling) without over-engineering.

**Next Step**: Implement Phase 1 with focus on robust alternating optimization starting from full AAA_LS initialization.

---

*Revised based on consensus from o3, Gemini Pro, and user feedback*  
*Prioritizes practical engineering over theoretical complexity*  
*Ready for streamlined implementation*