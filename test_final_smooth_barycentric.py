#!/usr/bin/env python3
"""
Test the final optimized smooth barycentric evaluation with W=0.01
Then test integration with AAA_FullOpt to see if it solves optimization issues
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from scipy.optimize import minimize
config.update("jax_enable_x64", True)

def naive_barycentric_eval(z, x, f, w):
    """Standard barycentric evaluation"""
    diffs = z - x
    weights = w / diffs
    num = jnp.sum(weights * f)
    den = jnp.sum(weights)
    return num / den

def smooth_barycentric_eval_final(z, x, f, w, W=0.01):
    """Final optimized smooth barycentric evaluation with W=0.01"""
    d = z - x
    d_sq = d**2
    alpha = jnp.tanh(d_sq / W)
    
    safe_far_term = jnp.nan_to_num(alpha / d, nan=0.0)
    N_far_unscaled = jnp.sum(safe_far_term * w * f)
    D_far_unscaled = jnp.sum(safe_far_term * w)
    
    one_minus_alpha = 1.0 - alpha
    N_close = jnp.sum(one_minus_alpha * w * f)
    D_close = jnp.sum(one_minus_alpha * w)
    
    d_scale = jnp.sum(one_minus_alpha * d)
    gamma = jnp.prod(alpha)
    
    N_final = N_close + d_scale * N_far_unscaled + gamma * N_far_unscaled
    D_final = D_close + d_scale * D_far_unscaled + gamma * D_far_unscaled
    
    return N_final / (D_final + 1e-30)

def test_accuracy_final():
    """Test final accuracy with W=0.01"""
    print("TESTING FINAL ACCURACY (W=0.01)")
    print("="*50)
    
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = jnp.array([0.0, 1.0, 4.0, 9.0, 16.0])
    w = jnp.ones(5)
    
    z_test = jnp.array([0.5, 1.5, 2.5, 3.5, -0.5, 4.5])
    
    max_error = 0
    print("Accuracy comparison:")
    for z in z_test:
        naive_val = naive_barycentric_eval(z, x, f, w)
        smooth_val = smooth_barycentric_eval_final(z, x, f, w)
        error = abs(naive_val - smooth_val)
        rel_error = error / abs(naive_val) * 100 if naive_val != 0 else 0
        max_error = max(max_error, error)
        
        print(f"z = {z:5.1f}: naive = {naive_val:12.6f}, smooth = {smooth_val:12.6f}")
        print(f"         error = {error:8.6f} ({rel_error:6.3f}%)")
        
    print(f"\nMaximum absolute error: {max_error:.8f}")
    if max_error < 1e-6:
        print("✅ EXCELLENT ACCURACY!")
    elif max_error < 1e-3:
        print("✅ Good accuracy")
    else:
        print("⚠️ Accuracy could be better")
    print()

def test_derivatives_final():
    """Test derivative smoothness"""
    print("TESTING DERIVATIVE SMOOTHNESS")
    print("="*50)
    
    x = jnp.array([0.0, 1.0, 2.0])
    f = jnp.array([1.0, 2.0, 4.0])
    w = jnp.array([1.0, 1.0, 1.0])
    
    def eval_func(z):
        return smooth_barycentric_eval_final(z, x, f, w)
    
    # Test around transition regions
    z_test = jnp.linspace(0.8, 1.2, 21)  # Around x[1] = 1.0
    
    print("Testing derivatives around x=1.0:")
    all_finite = True
    for z in z_test:
        try:
            val = eval_func(z)
            grad1 = jax.grad(eval_func)(z)
            grad2 = jax.grad(jax.grad(eval_func))(z)
            
            finite_all = jnp.all(jnp.isfinite([val, grad1, grad2]))
            if not finite_all:
                all_finite = False
                
            if z in [0.9, 1.0, 1.1]:  # Sample points
                print(f"z = {z:5.2f}: f = {val:8.4f}, f' = {grad1:8.4f}, f'' = {grad2:8.4f}, finite = {finite_all}")
                
        except Exception as e:
            print(f"z = {z:5.2f}: Failed - {e}")
            all_finite = False
    
    if all_finite:
        print("✅ ALL DERIVATIVES FINITE AND SMOOTH!")
    else:
        print("❌ Some derivative issues")
    print()

def test_aaa_optimization_simulation():
    """Simulate AAA_FullOpt optimization to see if it converges"""
    print("TESTING AAA OPTIMIZATION SIMULATION")
    print("="*50)
    
    # Create synthetic data
    t_data = jnp.linspace(0, 2*jnp.pi, 20)
    y_data = jnp.sin(t_data) + 0.01 * jnp.random.RandomState(42).randn(len(t_data))
    
    print(f"Data points: {len(t_data)}")
    print(f"Data range: [{t_data.min():.2f}, {t_data.max():.2f}]")
    
    # Initial guess for AAA parameters (support points, values, weights)
    m = 5  # Number of support points
    
    # Initialize support points spread across data range
    zj_init = jnp.linspace(t_data.min(), t_data.max(), m)
    fj_init = jnp.interp(zj_init, t_data, y_data)  # Interpolate initial values
    wj_init = jnp.ones(m)
    
    print(f"Initial support points: {zj_init}")
    print(f"Initial values: {fj_init}")
    
    def objective_function(params):
        """Objective function for AAA optimization using smooth evaluation"""
        zj, fj, wj = jnp.split(params, 3)
        
        # Evaluate at data points using smooth formula
        def eval_single(t):
            return smooth_barycentric_eval_final(t, zj, fj, wj)
        
        y_pred = jax.vmap(eval_single)(t_data)
        
        # Mean squared error
        mse = jnp.mean((y_data - y_pred)**2)
        
        # Light regularization to prevent extreme values
        reg = 1e-6 * (jnp.sum(wj**2) + jnp.sum((fj - jnp.mean(fj))**2))
        
        return mse + reg
    
    # Test gradient computation
    print("Testing gradient computation...")
    initial_params = jnp.concatenate([zj_init, fj_init, wj_init])
    
    try:
        initial_loss = objective_function(initial_params)
        grad = jax.grad(objective_function)(initial_params)
        
        grad_finite = jnp.all(jnp.isfinite(grad))
        grad_nonzero = jnp.any(jnp.abs(grad) > 1e-12)
        
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Gradient finite: {grad_finite}")
        print(f"Gradient non-zero: {grad_nonzero}")
        print(f"Gradient norm: {jnp.linalg.norm(grad):.6f}")
        
        if grad_finite and grad_nonzero:
            print("✅ GRADIENT COMPUTATION SUCCESSFUL!")
            
            # Try a few optimization steps
            print("\nTesting optimization steps...")
            
            def obj_for_scipy(params):
                return float(objective_function(params))
            
            def grad_for_scipy(params):
                return np.array(jax.grad(objective_function)(params))
            
            # Run a short optimization
            result = minimize(
                obj_for_scipy,
                initial_params,
                method='L-BFGS-B',
                jac=grad_for_scipy,
                options={'maxiter': 10, 'disp': True}
            )
            
            print(f"Optimization result: {result.success}")
            print(f"Final loss: {result.fun:.6f}")
            print(f"Loss improvement: {initial_loss - result.fun:.6f}")
            
            if result.success:
                print("🎉 OPTIMIZATION CONVERGED SUCCESSFULLY!")
            else:
                print("⚠️ Optimization had issues")
                
        else:
            print("❌ Gradient computation failed")
            
    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
    
    print()

if __name__ == "__main__":
    print("="*70)
    print("TESTING FINAL OPTIMIZED SMOOTH BARYCENTRIC EVALUATION")
    print("="*70)
    print()
    
    test_accuracy_final()
    test_derivatives_final()
    test_aaa_optimization_simulation()
    
    print("="*70)
    print("FINAL TESTING COMPLETE")
    print("="*70)