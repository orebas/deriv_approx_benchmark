#!/usr/bin/env python3
"""
Quick diagnostic script focused on AAA_FullOpt instability.
Tests specific high-noise scenarios where failures are likely.
"""

import numpy as np
import pandas as pd
from comprehensive_methods_library import AAA_FullOpt_Approximator, AAALeastSquaresApproximator, barycentric_eval
import json
import os
import jax
import jax.numpy as jnp
from scipy.interpolate import AAA

# Configure JAX
from jax import config
config.update("jax_enable_x64", True)

def create_test_data(noise_level=0.1, n_points=51):
    """Create simple test data with known derivatives"""
    t = np.linspace(0, 2*np.pi, n_points)
    # Use a function with known derivatives
    y_clean = np.sin(2*t) + 0.5*np.cos(3*t)
    d1_clean = 2*np.cos(2*t) - 1.5*np.sin(3*t)
    d2_clean = -4*np.sin(2*t) - 4.5*np.cos(3*t)
    
    # Add noise
    noise_scale = noise_level * np.std(y_clean)
    y_noisy = y_clean + noise_scale * np.random.randn(len(t))
    
    return t, y_noisy, y_clean, d1_clean, d2_clean

def diagnose_single_fit(t, y, method_name="AAA_FullOpt"):
    """Diagnose a single fit with detailed tracking"""
    
    if method_name == "AAA_FullOpt":
        approx = AAA_FullOpt_Approximator(t, y)
    else:
        approx = AAALeastSquaresApproximator(t, y)
    
    # Track what happens during fitting
    diagnostics = {
        'method': method_name,
        'data_stats': {
            'n_points': len(t),
            'y_std': float(np.std(y)),
            'y_range': float(np.max(y) - np.min(y))
        },
        'fit_stages': []
    }
    
    # Override the _fit_implementation to add tracking
    original_fit = approx._fit_implementation
    
    def tracked_fit():
        best_model = {'bic': np.inf, 'params': None, 'm': 0}
        n_data_points = len(approx.t)
        max_possible_m = min(20, n_data_points // 4) if method_name == "AAA_FullOpt" else min(25, n_data_points // 3)
        
        y_scale = jnp.std(approx.y)
        lambda_reg = 1e-4 * y_scale if y_scale > 1e-9 else 1e-4
        
        for m_target in range(3, max_possible_m):
            stage_info = {'m_target': m_target}
            
            try:
                aaa_obj = AAA(approx.t, approx.y, max_terms=m_target)
                zj_initial = jnp.array(aaa_obj.support_points)
                fj_initial = jnp.array(aaa_obj.support_values)
                wj_initial = jnp.array(aaa_obj.weights)
                
                m_actual = len(zj_initial)
                stage_info['m_actual'] = m_actual
                stage_info['initial_weight_condition'] = float(np.max(np.abs(wj_initial)) / (np.min(np.abs(wj_initial)) + 1e-12))
                
                if m_actual <= best_model.get('m', 0):
                    stage_info['skipped'] = True
                    diagnostics['fit_stages'].append(stage_info)
                    continue
                    
            except Exception as e:
                stage_info['aaa_init_failed'] = str(e)
                diagnostics['fit_stages'].append(stage_info)
                continue
            
            # Define objective function based on method
            if method_name == "AAA_FullOpt":
                def objective_func(params):
                    zj, fj, wj = jnp.split(params, 3)
                    vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj, fj, wj))
                    y_pred = vmap_bary_eval(approx.t)
                    error_term = jnp.sum((approx.y - y_pred)**2)
                    
                    d2_func = jax.grad(jax.grad(lambda x: barycentric_eval(x, zj, fj, wj)))
                    d2_values = jax.vmap(d2_func)(approx.t)
                    smoothness_term = jnp.sum(d2_values**2)
                    return error_term + lambda_reg * smoothness_term
                
                initial_params = jnp.concatenate([zj_initial, fj_initial, wj_initial])
                n_params = 3 * m_actual
            else:  # AAA_LS
                def objective_func(params):
                    fj, wj = jnp.split(params, 2)
                    vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj_initial, fj, wj))
                    y_pred = vmap_bary_eval(approx.t)
                    error_term = jnp.sum((approx.y - y_pred)**2)
                    
                    d2_func = jax.grad(jax.grad(lambda x: barycentric_eval(x, zj_initial, fj, wj)))
                    d2_values = jax.vmap(d2_func)(approx.t)
                    smoothness_term = jnp.sum(d2_values**2)
                    return error_term + lambda_reg * smoothness_term
                
                initial_params = jnp.concatenate([fj_initial, wj_initial])
                n_params = 2 * m_actual
            
            objective_with_grad = jax.jit(jax.value_and_grad(objective_func))
            
            # Test initial objective
            try:
                init_val, init_grad = objective_with_grad(initial_params)
                stage_info['initial_objective'] = float(init_val)
                stage_info['initial_grad_norm'] = float(jnp.linalg.norm(init_grad))
                
                if jnp.isnan(init_val) or jnp.any(jnp.isnan(init_grad)):
                    stage_info['initial_nan'] = True
                    diagnostics['fit_stages'].append(stage_info)
                    continue
                    
            except Exception as e:
                stage_info['objective_failed'] = str(e)
                diagnostics['fit_stages'].append(stage_info)
                continue
            
            def scipy_objective(params_flat):
                val, grad = objective_with_grad(params_flat)
                if jnp.isnan(val) or jnp.any(jnp.isnan(grad)):
                    return np.inf, np.zeros_like(params_flat)
                return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)
            
            try:
                from scipy.optimize import minimize
                result = minimize(
                    scipy_objective,
                    initial_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                stage_info['opt_success'] = result.success
                stage_info['opt_message'] = result.message
                stage_info['opt_iterations'] = result.nit
                stage_info['final_objective'] = float(result.fun)
                
                if result.success:
                    # Calculate BIC
                    if method_name == "AAA_FullOpt":
                        zj_final, fj_final, wj_final = jnp.split(result.x, 3)
                    else:
                        fj_final, wj_final = jnp.split(result.x, 2)
                        zj_final = zj_initial
                        
                    y_pred_final = jax.vmap(lambda x: barycentric_eval(x, zj_final, fj_final, wj_final))(approx.t)
                    pure_rss = jnp.sum((approx.y - y_pred_final)**2)
                    
                    k = n_params
                    bic = k * np.log(n_data_points) + n_data_points * np.log(pure_rss / n_data_points + 1e-12)
                    
                    stage_info['bic'] = float(bic)
                    stage_info['rss'] = float(pure_rss)
                    stage_info['final_weight_condition'] = float(np.max(np.abs(wj_final)) / (np.min(np.abs(wj_final)) + 1e-12))
                    
                    if method_name == "AAA_FullOpt":
                        stage_info['final_min_support_dist'] = float(np.min(np.diff(np.sort(zj_final))))
                    
                    if bic < best_model['bic']:
                        best_model.update({'bic': bic, 'params': result.x, 'm': m_actual})
                        stage_info['selected'] = True
                        
            except Exception as e:
                stage_info['optimization_error'] = str(e)
                
            diagnostics['fit_stages'].append(stage_info)
        
        diagnostics['best_model_m'] = best_model.get('m', 0)
        diagnostics['best_model_bic'] = float(best_model['bic']) if best_model['bic'] < np.inf else None
        
        # Set the approximator's parameters
        if best_model['params'] is not None:
            if method_name == "AAA_FullOpt":
                approx.zj, approx.fj, approx.wj = jnp.split(best_model['params'], 3)
            else:
                approx.fj, approx.wj = jnp.split(best_model['params'], 2)
                approx.zj = best_model.get('zj', zj_initial)  # Use the support points from best model
                
            def single_eval(x):
                return barycentric_eval(x, approx.zj, approx.fj, approx.wj)
                
            approx.ad_derivatives = [jax.jit(single_eval)]
            for _ in range(approx.max_derivative_supported):
                approx.ad_derivatives.append(jax.jit(jax.grad(approx.ad_derivatives[-1])))
            approx.success = True
        else:
            approx.success = False
    
    # Replace the method and fit
    approx._fit_implementation = tracked_fit
    
    try:
        approx.fit()
        diagnostics['fit_success'] = approx.success
        diagnostics['fit_time'] = approx.fit_time
        
        if approx.success:
            # Test evaluation
            try:
                result = approx.evaluate(t, max_derivative=2)
                diagnostics['eval_success'] = True
                
                # Check for extreme values
                diagnostics['y_max'] = float(np.max(np.abs(result['y'])))
                diagnostics['d1_max'] = float(np.max(np.abs(result['d1'])))
                diagnostics['d2_max'] = float(np.max(np.abs(result['d2'])))
                
            except Exception as e:
                diagnostics['eval_success'] = False
                diagnostics['eval_error'] = str(e)
        else:
            diagnostics['eval_success'] = False
            
    except Exception as e:
        diagnostics['fit_success'] = False
        diagnostics['fit_error'] = str(e)
        
    return diagnostics


def main():
    """Run focused diagnostic tests"""
    
    print("Running focused AAA stability diagnostics...\n")
    
    # Test configurations - focus on problematic cases
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    n_points_list = [31, 51, 101]
    
    results = []
    
    for noise in noise_levels:
        for n_points in n_points_list:
            print(f"Testing with noise={noise}, n_points={n_points}")
            
            # Create test data
            t, y_noisy, y_clean, d1_clean, d2_clean = create_test_data(noise, n_points)
            
            # Test AAA_FullOpt
            diag_fullopt = diagnose_single_fit(t, y_noisy, "AAA_FullOpt")
            diag_fullopt['noise_level'] = noise
            diag_fullopt['n_points'] = n_points
            results.append(diag_fullopt)
            
            # Test AAA_LS
            diag_ls = diagnose_single_fit(t, y_noisy, "AAA_LS")
            diag_ls['noise_level'] = noise
            diag_ls['n_points'] = n_points
            results.append(diag_ls)
            
            # Print quick summary
            fullopt_success = diag_fullopt.get('fit_success', False)
            ls_success = diag_ls.get('fit_success', False)
            print(f"  AAA_FullOpt: {'SUCCESS' if fullopt_success else 'FAILED'}")
            print(f"  AAA_LS: {'SUCCESS' if ls_success else 'FAILED'}")
            
            if fullopt_success and 'fit_stages' in diag_fullopt:
                # Find the selected model
                selected = [s for s in diag_fullopt['fit_stages'] if s.get('selected', False)]
                if selected:
                    s = selected[0]
                    print(f"    FullOpt - m={s.get('m_actual')}, weight_cond={s.get('final_weight_condition', 'N/A'):.2e}")
                    if 'final_min_support_dist' in s:
                        print(f"    FullOpt - min_support_dist={s['final_min_support_dist']:.2e}")
                        
            print()
    
    # Save results
    os.makedirs("diagnostic_results", exist_ok=True)
    with open("diagnostic_results/quick_aaa_diagnostic.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    # Analyze patterns
    print("\nSUMMARY ANALYSIS")
    print("="*60)
    
    # Success rates
    fullopt_results = [r for r in results if r['method'] == 'AAA_FullOpt']
    ls_results = [r for r in results if r['method'] == 'AAA_LS']
    
    fullopt_success_rate = sum(r.get('fit_success', False) for r in fullopt_results) / len(fullopt_results)
    ls_success_rate = sum(r.get('fit_success', False) for r in ls_results) / len(ls_results)
    
    print(f"Overall success rates:")
    print(f"  AAA_FullOpt: {fullopt_success_rate:.2%}")
    print(f"  AAA_LS: {ls_success_rate:.2%}")
    
    # Failure analysis
    print("\nAAA_FullOpt failure patterns:")
    failures = [r for r in fullopt_results if not r.get('fit_success', False)]
    
    for f in failures:
        print(f"\n  Noise={f['noise_level']}, n_points={f['n_points']}:")
        if 'fit_stages' in f:
            # Check if any stage had issues
            for stage in f['fit_stages']:
                if 'initial_nan' in stage:
                    print(f"    - NaN in initial objective at m={stage['m_target']}")
                if 'initial_weight_condition' in stage and stage['initial_weight_condition'] > 1e10:
                    print(f"    - Extreme weight condition: {stage['initial_weight_condition']:.2e} at m={stage['m_target']}")
                if 'opt_success' in stage and not stage['opt_success']:
                    print(f"    - Optimization failed at m={stage['m_target']}: {stage.get('opt_message', 'unknown')}")
        
        if f.get('best_model_m', 0) == 0:
            print(f"    - No viable model found")
            
    print(f"\nDetailed results saved to: diagnostic_results/quick_aaa_diagnostic.json")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()