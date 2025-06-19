#!/usr/bin/env python3
"""
Diagnostic script to investigate AAA_FullOpt stability issues.
Focuses on:
1. Tracking optimization convergence
2. Identifying failure patterns at high noise/derivatives
3. Comparing with AAA_LS stability
4. Collecting detailed debug information
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from comprehensive_methods_library import AAA_FullOpt_Approximator, AAALeastSquaresApproximator
import traceback
import json
from datetime import datetime
import os
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
from scipy.interpolate import AAA

# Configure JAX
from jax import config
config.update("jax_enable_x64", True)

class DiagnosticAAA_FullOpt(AAA_FullOpt_Approximator):
    """Extended AAA_FullOpt with diagnostic tracking"""
    
    def __init__(self, t, y, name="AAA_FullOpt_Diag"):
        super().__init__(t, y, name)
        self.diagnostics = {
            'optimization_history': [],
            'model_selection': [],
            'failure_reasons': [],
            'parameter_evolution': [],
            'condition_numbers': []
        }
        
    def _fit_implementation(self):
        best_model = {'bic': np.inf, 'params': None, 'm': 0}
        n_data_points = len(self.t)
        max_possible_m = min(20, n_data_points // 4)
        
        # Make regularization adaptive to data scale
        y_scale = jnp.std(self.y)
        lambda_reg = 1e-4 * y_scale if y_scale > 1e-9 else 1e-4
        
        for m_target in range(3, max_possible_m):
            try:
                aaa_obj = AAA(self.t, self.y, max_terms=m_target)
                zj_initial = jnp.array(aaa_obj.support_points)
                fj_initial = jnp.array(aaa_obj.support_values)
                wj_initial = jnp.array(aaa_obj.weights)
                
                m_actual = len(zj_initial)
                if m_actual <= best_model.get('m', 0):
                    continue
                    
                # Track initial configuration
                self.diagnostics['model_selection'].append({
                    'm_target': m_target,
                    'm_actual': m_actual,
                    'initial_support_points': zj_initial.tolist(),
                    'initial_max_weight': float(np.max(np.abs(wj_initial))),
                    'initial_min_weight': float(np.min(np.abs(wj_initial))),
                    'initial_condition': float(np.max(np.abs(wj_initial)) / (np.min(np.abs(wj_initial)) + 1e-12))
                })
                    
            except Exception as e:
                self.diagnostics['failure_reasons'].append({
                    'm_target': m_target,
                    'stage': 'AAA_initialization',
                    'error': str(e)
                })
                continue

            # Track optimization iterations
            iter_count = 0
            def callback_func(xk):
                nonlocal iter_count
                iter_count += 1
                zj_k, fj_k, wj_k = jnp.split(xk, 3)
                
                # Calculate condition number
                cond = float(np.max(np.abs(wj_k)) / (np.min(np.abs(wj_k)) + 1e-12))
                
                # Check for close support points
                min_dist = float(np.min(np.diff(np.sort(zj_k))))
                
                self.diagnostics['parameter_evolution'].append({
                    'm': m_actual,
                    'iteration': iter_count,
                    'weight_condition': cond,
                    'min_support_distance': min_dist,
                    'max_weight': float(np.max(np.abs(wj_k))),
                    'min_weight': float(np.min(np.abs(wj_k)))
                })

            def objective_func(params):
                zj, fj, wj = jnp.split(params, 3)
                
                # Modified to use barycentric_eval from comprehensive_methods_library
                from comprehensive_methods_library import barycentric_eval
                vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj, fj, wj))
                y_pred = vmap_bary_eval(self.t)
                error_term = jnp.sum((self.y - y_pred)**2)
                
                d2_func = jax.grad(jax.grad(lambda x: barycentric_eval(x, zj, fj, wj)))
                d2_values = jax.vmap(d2_func)(self.t)
                smoothness_term = jnp.sum(d2_values**2)
                return error_term + lambda_reg * smoothness_term

            objective_with_grad = jax.jit(jax.value_and_grad(objective_func))

            def scipy_objective(params_flat):
                val, grad = objective_with_grad(params_flat)
                if jnp.isnan(val) or jnp.any(jnp.isnan(grad)):
                    self.diagnostics['failure_reasons'].append({
                        'm': m_actual,
                        'stage': 'optimization',
                        'error': 'NaN in objective or gradient'
                    })
                    return np.inf, np.zeros_like(params_flat)
                return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)

            initial_params = jnp.concatenate([zj_initial, fj_initial, wj_initial])
            
            try:
                result = minimize(
                    scipy_objective,
                    initial_params,
                    method='L-BFGS-B',
                    jac=True,
                    callback=callback_func,
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                self.diagnostics['optimization_history'].append({
                    'm': m_actual,
                    'success': result.success,
                    'message': result.message,
                    'n_iterations': result.nit,
                    'final_objective': float(result.fun)
                })
                
                if not result.success: 
                    self.diagnostics['failure_reasons'].append({
                        'm': m_actual,
                        'stage': 'optimization_convergence',
                        'error': result.message
                    })
                    continue
                
                final_params = result.x
                
                # Re-evaluate the error term (RSS) without the penalty terms
                zj_final, fj_final, wj_final = jnp.split(final_params, 3)
                from comprehensive_methods_library import barycentric_eval
                y_pred_final = jax.vmap(lambda x: barycentric_eval(x, zj_final, fj_final, wj_final))(self.t)
                pure_rss = jnp.sum((self.y - y_pred_final)**2)

                k = 3 * m_actual
                bic = k * np.log(n_data_points) + n_data_points * np.log(pure_rss / n_data_points + 1e-12)

                if bic < best_model['bic']:
                    best_model.update({'bic': bic, 'params': final_params, 'm': m_actual})
                    
            except Exception as e:
                self.diagnostics['failure_reasons'].append({
                    'm': m_actual,
                    'stage': 'optimization_execution',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                continue

        if best_model['params'] is None:
            self.success = False
            self.diagnostics['failure_reasons'].append({
                'stage': 'final',
                'error': 'No viable model found'
            })
            return

        self.zj, self.fj, self.wj = jnp.split(best_model['params'], 3)
        
        # Store final diagnostic info
        self.diagnostics['final_model'] = {
            'm': best_model['m'],
            'bic': float(best_model['bic']),
            'weight_condition': float(np.max(np.abs(self.wj)) / (np.min(np.abs(self.wj)) + 1e-12)),
            'min_support_distance': float(np.min(np.diff(np.sort(self.zj))))
        }
        
        from comprehensive_methods_library import barycentric_eval
        def single_eval(x):
            return barycentric_eval(x, self.zj, self.fj, self.wj)
            
        self.ad_derivatives = [jax.jit(single_eval)]
        for _ in range(self.max_derivative_supported):
            self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))


def run_diagnostic_test(ode_name, noise_level, max_derivative=4):
    """Run diagnostic test on a specific ODE and noise level"""
    
    # Load test data
    data_path = f"test_data/{ode_name}/noise_{noise_level}"
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found")
        return None
        
    noisy_df = pd.read_csv(f"{data_path}/noisy_data.csv")
    truth_df = pd.read_csv(f"{data_path}/truth_data.csv")
    
    t = noisy_df['t'].values
    
    results = []
    
    # Test all observables
    for col in noisy_df.columns:
        if col == 't':
            continue
            
        y = noisy_df[col].values
        
        # Test AAA_FullOpt with diagnostics
        print(f"\nTesting AAA_FullOpt on {ode_name}/{col} with noise={noise_level}")
        
        diag_fullopt = DiagnosticAAA_FullOpt(t, y)
        try:
            diag_fullopt.fit()
            eval_result = diag_fullopt.evaluate(t, max_derivative=max_derivative)
            
            # Calculate errors
            errors = {}
            if col in truth_df.columns:
                errors['y'] = np.sqrt(np.mean((eval_result['y'] - truth_df[col].values)**2))
                
            for d in range(1, max_derivative + 1):
                deriv_col = f"d{d}_{col}"
                if deriv_col in truth_df.columns:
                    errors[f'd{d}'] = np.sqrt(np.mean((eval_result[f'd{d}'] - truth_df[deriv_col].values)**2))
                    
            result = {
                'ode': ode_name,
                'observable': col,
                'noise_level': noise_level,
                'method': 'AAA_FullOpt',
                'success': diag_fullopt.success,
                'errors': errors,
                'diagnostics': diag_fullopt.diagnostics
            }
            
        except Exception as e:
            result = {
                'ode': ode_name,
                'observable': col,
                'noise_level': noise_level,
                'method': 'AAA_FullOpt',
                'success': False,
                'errors': {},
                'diagnostics': {
                    'failure_reasons': [{
                        'stage': 'top_level',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }]
                }
            }
            
        results.append(result)
        
        # Also test AAA_LS for comparison
        print(f"Testing AAA_LS on {ode_name}/{col} with noise={noise_level}")
        
        aaa_ls = AAALeastSquaresApproximator(t, y)
        try:
            aaa_ls.fit()
            eval_result = aaa_ls.evaluate(t, max_derivative=max_derivative)
            
            # Calculate errors
            errors = {}
            if col in truth_df.columns:
                errors['y'] = np.sqrt(np.mean((eval_result['y'] - truth_df[col].values)**2))
                
            for d in range(1, max_derivative + 1):
                deriv_col = f"d{d}_{col}"
                if deriv_col in truth_df.columns:
                    errors[f'd{d}'] = np.sqrt(np.mean((eval_result[f'd{d}'] - truth_df[deriv_col].values)**2))
                    
            result = {
                'ode': ode_name,
                'observable': col,
                'noise_level': noise_level,
                'method': 'AAA_LS',
                'success': aaa_ls.success,
                'errors': errors,
                'diagnostics': {}
            }
            
        except Exception as e:
            result = {
                'ode': ode_name,
                'observable': col,
                'noise_level': noise_level,
                'method': 'AAA_LS',
                'success': False,
                'errors': {},
                'diagnostics': {}
            }
            
        results.append(result)
        
    return results


def analyze_diagnostic_results(results):
    """Analyze diagnostic results to identify failure patterns"""
    
    analysis = {
        'failure_patterns': {},
        'stability_comparison': {},
        'parameter_issues': []
    }
    
    for result in results:
        if not result['success'] and result['method'] == 'AAA_FullOpt':
            # Analyze failure reasons
            if 'diagnostics' in result and 'failure_reasons' in result['diagnostics']:
                for failure in result['diagnostics']['failure_reasons']:
                    stage = failure.get('stage', 'unknown')
                    if stage not in analysis['failure_patterns']:
                        analysis['failure_patterns'][stage] = []
                    analysis['failure_patterns'][stage].append({
                        'ode': result['ode'],
                        'noise': result['noise_level'],
                        'error': failure.get('error', 'unknown')
                    })
                    
            # Check for parameter evolution issues
            if 'diagnostics' in result and 'parameter_evolution' in result['diagnostics']:
                evol = result['diagnostics']['parameter_evolution']
                if evol:
                    # Check for extreme condition numbers
                    max_cond = max(p['weight_condition'] for p in evol)
                    min_dist = min(p['min_support_distance'] for p in evol)
                    
                    if max_cond > 1e10 or min_dist < 1e-6:
                        analysis['parameter_issues'].append({
                            'ode': result['ode'],
                            'noise': result['noise_level'],
                            'max_condition': max_cond,
                            'min_support_distance': min_dist
                        })
                        
    return analysis


def main():
    """Run comprehensive diagnostic tests"""
    
    # Test configurations
    ode_problems = ["lv_periodic", "sir", "vanderpol"]
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    
    all_results = []
    
    for ode in ode_problems:
        for noise in noise_levels:
            print(f"\n{'='*60}")
            print(f"Testing {ode} with noise level {noise}")
            print('='*60)
            
            results = run_diagnostic_test(ode, noise)
            if results:
                all_results.extend(results)
                
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"diagnostic_results/aaa_stability_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/raw_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
        
    # Analyze results
    analysis = analyze_diagnostic_results(all_results)
    
    with open(f"{output_dir}/analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
        
    # Create summary report
    summary = []
    for result in all_results:
        summary.append({
            'ode': result['ode'],
            'observable': result['observable'],
            'noise_level': result['noise_level'],
            'method': result['method'],
            'success': result['success'],
            'y_error': result['errors'].get('y', np.nan) if result['errors'] else np.nan,
            'd1_error': result['errors'].get('d1', np.nan) if result['errors'] else np.nan,
            'd2_error': result['errors'].get('d2', np.nan) if result['errors'] else np.nan
        })
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Success rates by method and noise level
    success_rates = summary_df.groupby(['method', 'noise_level'])['success'].mean()
    print("\nSuccess Rates:")
    print(success_rates)
    
    # Average errors for successful runs
    print("\nAverage RMSE for successful runs:")
    successful = summary_df[summary_df['success']]
    error_summary = successful.groupby(['method', 'noise_level'])[['y_error', 'd1_error', 'd2_error']].mean()
    print(error_summary)
    
    print(f"\nDetailed results saved to: {output_dir}/")
    

if __name__ == "__main__":
    main()