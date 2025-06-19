#!/usr/bin/env python3
"""
Minimal test to demonstrate AAA_FullOpt instability issues
"""

import numpy as np
from scipy.interpolate import AAA
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

# Import the barycentric_eval function
from comprehensive_methods_library import barycentric_eval

def test_aaa_stability():
    """Simple test showing AAA_FullOpt stability issues"""
    
    # Create simple noisy data
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 31)
    y_clean = np.sin(t)
    
    # Test with different noise levels
    noise_levels = [0.01, 0.05, 0.1]
    
    for noise in noise_levels:
        print(f"\nTesting with noise level: {noise}")
        y_noisy = y_clean + noise * np.random.randn(len(t))
        
        # Get AAA approximation
        try:
            aaa_obj = AAA(t, y_noisy, max_terms=10)
            zj = jnp.array(aaa_obj.support_points)
            fj = jnp.array(aaa_obj.support_values)
            wj = jnp.array(aaa_obj.weights)
            
            print(f"  Initial AAA: m={len(zj)}")
            print(f"  Weight condition: {np.max(np.abs(wj)) / (np.min(np.abs(wj)) + 1e-12):.2e}")
            print(f"  Min support distance: {np.min(np.diff(np.sort(zj))):.2e}")
            
            # Test AAA_LS approach (fixed support points)
            def objective_ls(params):
                fj_opt, wj_opt = jnp.split(params, 2)
                vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj, fj_opt, wj_opt))
                y_pred = vmap_bary_eval(t)
                return jnp.sum((y_noisy - y_pred)**2)
            
            # Test AAA_FullOpt approach (all parameters free)
            def objective_full(params):
                zj_opt, fj_opt, wj_opt = jnp.split(params, 3)
                vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj_opt, fj_opt, wj_opt))
                y_pred = vmap_bary_eval(t)
                return jnp.sum((y_noisy - y_pred)**2)
            
            # Evaluate gradients at initial point
            grad_ls = jax.grad(objective_ls)
            grad_full = jax.grad(objective_full)
            
            params_ls = jnp.concatenate([fj, wj])
            params_full = jnp.concatenate([zj, fj, wj])
            
            try:
                g_ls = grad_ls(params_ls)
                print(f"  AAA_LS gradient norm: {jnp.linalg.norm(g_ls):.2e}")
            except Exception as e:
                print(f"  AAA_LS gradient failed: {type(e).__name__}")
                
            try:
                g_full = grad_full(params_full)
                print(f"  AAA_FullOpt gradient norm: {jnp.linalg.norm(g_full):.2e}")
                
                # Check gradient components
                gz, gf, gw = jnp.split(g_full, 3)
                print(f"    Support point grad norm: {jnp.linalg.norm(gz):.2e}")
                print(f"    Function value grad norm: {jnp.linalg.norm(gf):.2e}")
                print(f"    Weight grad norm: {jnp.linalg.norm(gw):.2e}")
                
            except Exception as e:
                print(f"  AAA_FullOpt gradient failed: {type(e).__name__}")
                
            # Test with slightly perturbed support points
            print("\n  Testing perturbation sensitivity:")
            for eps in [1e-6, 1e-4, 1e-2]:
                zj_pert = zj + eps * np.random.randn(len(zj))
                params_pert = jnp.concatenate([zj_pert, fj, wj])
                try:
                    obj_pert = objective_full(params_pert)
                    print(f"    eps={eps:.0e}: obj={obj_pert:.2e}")
                except:
                    print(f"    eps={eps:.0e}: FAILED")
                    
        except Exception as e:
            print(f"  AAA initialization failed: {e}")


def analyze_support_point_clustering():
    """Analyze how support points cluster with noise"""
    
    print("\nAnalyzing support point clustering:")
    
    np.random.seed(123)
    t = np.linspace(0, 2*np.pi, 51)
    y_clean = np.sin(2*t) + 0.3*np.cos(3*t)
    
    for noise in [0.01, 0.1]:
        y_noisy = y_clean + noise * np.random.randn(len(t))
        
        print(f"\n  Noise level: {noise}")
        for max_terms in [5, 10, 15]:
            try:
                aaa_obj = AAA(t, y_noisy, max_terms=max_terms)
                zj = np.array(aaa_obj.support_points)
                
                # Check for clustering
                distances = np.diff(np.sort(zj))
                min_dist = np.min(distances)
                mean_dist = np.mean(distances)
                
                print(f"    m={len(zj)}: min_dist={min_dist:.2e}, mean_dist={mean_dist:.2e}, ratio={mean_dist/min_dist:.1f}")
                
            except:
                print(f"    m_target={max_terms}: FAILED")


if __name__ == "__main__":
    test_aaa_stability()
    analyze_support_point_clustering()