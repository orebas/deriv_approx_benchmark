#!/usr/bin/env python3
"""
Stable AAA_FullOpt implementation using freeze-and-relax strategy.

Based on insights from Remez algorithm and Vector Fitting:
1. Alternate between optimizing (fj, wj) with fixed zj
2. Then carefully update zj with constraints
3. This avoids the simultaneous optimization that causes instabilities
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.interpolate import AAA
from comprehensive_methods_library import smooth_barycentric_eval

class StableAAA_FullOpt_Approximator:
    """
    Stable version of AAA_FullOpt using alternating optimization.
    
    Key improvements:
    1. Freezes zj while optimizing fj/wj (linear-like problem)
    2. Updates zj with constraints to prevent pole collisions
    3. Uses barrier function to keep poles away from data points
    """
    
    def __init__(self, t, y, name="AAA_FullOpt_Stable"):
        self.t = jnp.array(t)
        self.y = jnp.array(y)
        self.name = name
        self.max_derivative_supported = 7
        self.zj = None
        self.fj = None
        self.wj = None
        self.fitted = False
        self.success = True
        
    def _fit_implementation(self):
        """Fit using stable alternating optimization."""
        
        best_model = {'bic': np.inf, 'zj': None, 'fj': None, 'wj': None}
        n_data_points = len(self.t)
        max_possible_m = min(20, n_data_points // 4)
        
        # Data bounds for constraining zj
        t_min, t_max = self.t.min(), self.t.max()
        t_range = t_max - t_min
        # Allow zj slightly outside data range
        z_bounds = (t_min - 0.1 * t_range, t_max + 0.1 * t_range)
        
        for m_target in range(3, max_possible_m):
            try:
                # Step 1: Get initial zj from standard AAA
                aaa_obj = AAA(self.t, self.y, max_terms=m_target)
                zj_init = jnp.array(aaa_obj.support_points)
                fj_init = jnp.array(aaa_obj.support_values)
                wj_init = jnp.array(aaa_obj.weights)
                m_actual = len(zj_init)
                
            except Exception:
                # Fallback: uniform spacing
                indices = np.linspace(0, len(self.t) - 1, m_target, dtype=int)
                zj_init = self.t[indices]
                fj_init = self.y[indices]
                wj_init = jnp.ones(m_target)
                m_actual = m_target
            
            # Current parameters
            zj_current = zj_init.copy()
            fj_current = fj_init.copy()
            wj_current = wj_init.copy()
            
            # Alternating optimization
            n_outer_iterations = 5
            for outer_iter in range(n_outer_iterations):
                
                # Step 2: Fix zj, optimize fj and wj
                def objective_fj_wj(params):
                    fj, wj = jnp.split(params, 2)
                    # Use smooth evaluation for stability
                    vmap_eval = jax.vmap(lambda x: smooth_barycentric_eval(x, zj_current, fj, wj))
                    y_pred = vmap_eval(self.t)
                    error = jnp.sum((self.y - y_pred)**2)
                    # Light regularization on weights
                    reg_term = 1e-8 * jnp.sum(wj**2)
                    return error + reg_term
                
                # Optimize fj and wj
                initial_params = jnp.concatenate([fj_current, wj_current])
                result_fj_wj = minimize(
                    lambda p: float(objective_fj_wj(p)),
                    initial_params,
                    method='L-BFGS-B',
                    options={'maxiter': 1000}
                )
                
                if result_fj_wj.success:
                    fj_current, wj_current = jnp.split(result_fj_wj.x, 2)
                
                # Step 3: Fix fj/wj, carefully update zj
                if outer_iter < n_outer_iterations - 1:  # Don't update zj on last iteration
                    
                    def objective_zj_with_barrier(zj):
                        # Main error term
                        vmap_eval = jax.vmap(lambda x: smooth_barycentric_eval(x, zj, fj_current, wj_current))
                        y_pred = vmap_eval(self.t)
                        error = jnp.sum((self.y - y_pred)**2)
                        
                        # Barrier function to prevent pole collisions
                        barrier = 0.0
                        min_separation = 0.01 * t_range / m_actual
                        
                        # Penalize zj getting too close to data points
                        for i in range(len(zj)):
                            distances = jnp.abs(self.t - zj[i])
                            min_dist = jnp.min(distances)
                            barrier += 1.0 / (min_dist + 1e-6)**2
                        
                        # Penalize zj getting too close to each other
                        for i in range(len(zj)):
                            for j in range(i+1, len(zj)):
                                dist = jnp.abs(zj[i] - zj[j])
                                if dist < min_separation:
                                    barrier += 1.0 / (dist + 1e-6)**2
                        
                        # Scale barrier appropriately
                        barrier_weight = 1e-6 * error / (1.0 + barrier)
                        
                        return error + barrier_weight * barrier
                    
                    # Set up bounds for each zj
                    bounds = [(z_bounds[0], z_bounds[1]) for _ in range(m_actual)]
                    
                    # Optimize zj with constraints
                    result_zj = minimize(
                        lambda z: float(objective_zj_with_barrier(z)),
                        zj_current,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100}  # Limited iterations for stability
                    )
                    
                    if result_zj.success:
                        # Only accept the update if it improves things
                        new_error = objective_zj_with_barrier(result_zj.x)
                        old_error = objective_zj_with_barrier(zj_current)
                        if new_error < old_error:
                            zj_current = result_zj.x
            
            # Evaluate final model
            vmap_eval = jax.vmap(lambda x: smooth_barycentric_eval(x, zj_current, fj_current, wj_current))
            y_pred_final = vmap_eval(self.t)
            rss = jnp.sum((self.y - y_pred_final)**2)
            
            # BIC calculation
            k = 3 * m_actual  # zj, fj, wj parameters
            bic = k * np.log(n_data_points) + n_data_points * np.log(rss / n_data_points + 1e-12)
            
            if bic < best_model['bic']:
                best_model = {
                    'bic': bic,
                    'zj': zj_current,
                    'fj': fj_current,
                    'wj': wj_current
                }
        
        # Set final parameters
        if best_model['zj'] is not None:
            self.zj = best_model['zj']
            self.fj = best_model['fj'] 
            self.wj = best_model['wj']
            self.success = True
        else:
            self.success = False
            
    def fit(self):
        """Public fit method."""
        self._fit_implementation()
        self.fitted = True
        
        if self.success:
            # Build derivative functions
            def single_eval(x):
                return smooth_barycentric_eval(x, self.zj, self.fj, self.wj)
            
            self.ad_derivatives = [jax.jit(single_eval)]
            for _ in range(self.max_derivative_supported):
                self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))
    
    def evaluate(self, t_eval, max_derivative=5):
        """Evaluate function and derivatives."""
        if not self.fitted:
            self.fit()
            
        if not self.success:
            # Return NaN if fitting failed
            results = {'success': False}
            results['y'] = np.full_like(t_eval, np.nan)
            for d in range(1, max_derivative + 1):
                results[f'd{d}'] = np.full_like(t_eval, np.nan)
            return results
        
        results = {'success': True}
        
        # Evaluate function and derivatives
        results['y'] = np.array(jax.vmap(self.ad_derivatives[0])(t_eval))
        
        for d in range(1, min(max_derivative + 1, len(self.ad_derivatives))):
            try:
                results[f'd{d}'] = np.array(jax.vmap(self.ad_derivatives[d])(t_eval))
            except:
                results[f'd{d}'] = np.full_like(t_eval, np.nan)
                
        return results


if __name__ == "__main__":
    # Test the stable implementation
    print("Testing Stable AAA_FullOpt Implementation")
    print("=" * 50)
    
    # Create test data
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.001 * np.random.RandomState(42).randn(len(t))
    
    # Test stable version
    method = StableAAA_FullOpt_Approximator(t, y)
    method.fit()
    
    if method.success:
        print("✅ Fitting successful!")
        print(f"Number of support points: {len(method.zj)}")
        print(f"Support points range: [{method.zj.min():.3f}, {method.zj.max():.3f}]")
        print(f"Data range: [{t.min():.3f}, {t.max():.3f}]")
        
        # Test evaluation
        t_test = np.linspace(0.5, 5.5, 10)
        results = method.evaluate(t_test, max_derivative=7)
        
        print("\nDerivative evaluation results:")
        for order in range(8):
            key = 'y' if order == 0 else f'd{order}'
            if key in results:
                values = results[key]
                all_finite = np.all(np.isfinite(values))
                if all_finite:
                    max_abs = np.max(np.abs(values))
                    print(f"  Order {order}: ✅ All finite (max abs: {max_abs:.2e})")
                else:
                    finite_count = np.sum(np.isfinite(values))
                    print(f"  Order {order}: ⚠️  {finite_count}/{len(values)} finite")
    else:
        print("❌ Fitting failed!")