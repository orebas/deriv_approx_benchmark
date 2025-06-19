#!/usr/bin/env python3
"""
Stable AAA implementations with various fixes for the instability issues.
Based on the deep analysis of AAA_FullOpt problems.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from scipy.optimize import minimize
from scipy.interpolate import AAA
from comprehensive_methods_library import DerivativeApproximator, barycentric_eval

class AAA_TwoStage_Approximator(DerivativeApproximator):
    """
    Two-stage AAA: First run AAA_LS to get stable initialization,
    then optionally refine with limited AAA_FullOpt.
    """
    def __init__(self, t, y, name="AAA_TwoStage", enable_refinement=True, refinement_steps=100):
        super().__init__(t, y, name)
        self.max_derivative_supported = 5
        self.enable_refinement = enable_refinement
        self.refinement_steps = refinement_steps
        self.zj = None
        self.fj = None
        self.wj = None
        self.ad_derivatives = []
        self.success = True
        self.stage1_bic = None
        self.stage2_bic = None
        
    def _fit_implementation(self):
        """Two-stage fitting: AAA_LS followed by optional refinement"""
        
        # Stage 1: Run AAA_LS to get stable baseline
        stage1_success = self._run_aaa_ls_stage()
        
        if not stage1_success:
            self.success = False
            return
            
        # Stage 2: Optional refinement with constrained AAA_FullOpt
        if self.enable_refinement:
            self._run_refinement_stage()
        
        # Build derivative functions
        self._build_derivative_functions()
        
    def _run_aaa_ls_stage(self):
        """Stage 1: AAA_LS implementation"""
        best_model = {'bic': np.inf, 'params': None, 'zj': None, 'm': 0}
        n_data_points = len(self.t)
        max_possible_m = min(25, n_data_points // 3)
        
        y_scale = jnp.std(self.y)
        lambda_reg = 1e-4 * y_scale if y_scale > 1e-9 else 1e-4
        
        for m_target in range(3, max_possible_m):
            try:
                aaa_obj = AAA(self.t, self.y, max_terms=m_target)
                zj = jnp.array(aaa_obj.support_points)
                fj_initial = jnp.array(aaa_obj.support_values)
                wj_initial = jnp.array(aaa_obj.weights)
                
                m_actual = len(zj)
                if m_actual <= best_model.get('m', 0):
                    continue
            except Exception:
                continue
                
            # AAA_LS objective: optimize only fj and wj
            def objective_func(params):
                fj, wj = jnp.split(params, 2)
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
                    return np.inf, np.zeros_like(params_flat)
                return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)
                
            initial_params = jnp.concatenate([fj_initial, wj_initial])
            
            try:
                result = minimize(
                    scipy_objective,
                    initial_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if not result.success:
                    continue
                    
                final_params = result.x
                rss = result.fun
                k = 2 * m_actual
                bic = k * np.log(n_data_points) + n_data_points * np.log(rss / n_data_points + 1e-12)
                
                if bic < best_model['bic']:
                    best_model.update({
                        'bic': bic, 'params': final_params, 'zj': zj, 'm': m_actual
                    })
                    
            except Exception:
                continue
                
        if best_model['params'] is None:
            return False
            
        self.zj = best_model['zj']
        self.fj, self.wj = jnp.split(best_model['params'], 2)
        self.stage1_bic = best_model['bic']
        return True
        
    def _run_refinement_stage(self):
        """Stage 2: Limited refinement with small support point perturbations"""
        
        # Store stage 1 results as backup
        zj_stage1 = self.zj.copy()
        fj_stage1 = self.fj.copy()
        wj_stage1 = self.wj.copy()
        
        # Refinement parameters
        n_data_points = len(self.t)
        y_scale = jnp.std(self.y)
        lambda_reg = 1e-4 * y_scale if y_scale > 1e-9 else 1e-4
        
        # Calculate domain scale for perturbation limits
        domain_scale = float(jnp.max(self.t) - jnp.min(self.t))
        max_perturbation = 0.1 * domain_scale / len(self.zj)  # Conservative limit
        
        def refined_objective(params):
            zj_delta, fj, wj = jnp.split(params, 3)
            # Limit support point perturbations
            zj = self.zj + jnp.clip(zj_delta, -max_perturbation, max_perturbation)
            
            vmap_bary_eval = jax.vmap(lambda x: barycentric_eval(x, zj, fj, wj))
            y_pred = vmap_bary_eval(self.t)
            error_term = jnp.sum((self.y - y_pred)**2)
            
            # Enhanced regularization
            d2_func = jax.grad(jax.grad(lambda x: barycentric_eval(x, zj, fj, wj)))
            d2_values = jax.vmap(d2_func)(self.t)
            smoothness_term = jnp.sum(d2_values**2)
            
            # Add separation penalty
            zj_sorted = jnp.sort(zj)
            dists = jnp.diff(zj_sorted)
            min_dist_allowed = 1e-4 * domain_scale
            separation_penalty = jnp.sum(jax.nn.relu(min_dist_allowed - dists))
            
            return error_term + lambda_reg * smoothness_term + 1e-2 * separation_penalty
            
        objective_with_grad = jax.jit(jax.value_and_grad(refined_objective))
        
        def scipy_objective(params_flat):
            val, grad = objective_with_grad(params_flat)
            if jnp.isnan(val) or jnp.any(jnp.isnan(grad)):
                return np.inf, np.zeros_like(params_flat)
            return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)
            
        # Initialize with small perturbations
        zj_delta_init = jnp.zeros_like(self.zj)
        initial_params = jnp.concatenate([zj_delta_init, self.fj, self.wj])
        
        try:
            result = minimize(
                scipy_objective,
                initial_params,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': self.refinement_steps,
                    'ftol': 1e-10,
                    'gtol': 1e-6
                }
            )
            
            if result.success:
                # Extract refined parameters
                zj_delta_final, fj_final, wj_final = jnp.split(result.x, 3)
                zj_final = self.zj + jnp.clip(zj_delta_final, -max_perturbation, max_perturbation)
                
                # Calculate BIC for refined model
                y_pred_final = jax.vmap(lambda x: barycentric_eval(x, zj_final, fj_final, wj_final))(self.t)
                pure_rss = jnp.sum((self.y - y_pred_final)**2)
                k = 3 * len(self.zj)
                stage2_bic = k * np.log(n_data_points) + n_data_points * np.log(pure_rss / n_data_points + 1e-12)
                
                # Only accept if refinement improves BIC significantly
                if stage2_bic < self.stage1_bic - 2.0:  # Require meaningful improvement
                    self.zj = zj_final
                    self.fj = fj_final
                    self.wj = wj_final
                    self.stage2_bic = float(stage2_bic)
                else:
                    # Revert to stage 1 results
                    self.zj = zj_stage1
                    self.fj = fj_stage1
                    self.wj = wj_stage1
            else:
                # Revert to stage 1 results
                self.zj = zj_stage1
                self.fj = fj_stage1
                self.wj = wj_stage1
                
        except Exception:
            # Revert to stage 1 results on any error
            self.zj = zj_stage1
            self.fj = fj_stage1
            self.wj = wj_stage1
            
    def _build_derivative_functions(self):
        """Build JAX derivative functions"""
        def single_eval(x):
            return barycentric_eval(x, self.zj, self.fj, self.wj)
            
        self.ad_derivatives = [jax.jit(single_eval)]
        for _ in range(self.max_derivative_supported):
            self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))
            
    def _evaluate_function(self, t_eval):
        if not self.success:
            return np.full_like(t_eval, np.nan)
        return np.array(jax.vmap(self.ad_derivatives[0])(t_eval))
        
    def _evaluate_derivative(self, t_eval, order):
        if not self.success or order >= len(self.ad_derivatives):
            return np.full_like(t_eval, np.nan)
        return np.array(jax.vmap(self.ad_derivatives[order])(t_eval))


class AAA_SmoothBarycentric_Approximator(DerivativeApproximator):
    """
    AAA_FullOpt with smooth barycentric evaluation to fix gradient discontinuities
    """
    def __init__(self, t, y, name="AAA_SmoothBary", smooth_tolerance=1e-8):
        super().__init__(t, y, name)
        self.max_derivative_supported = 5
        self.smooth_tolerance = smooth_tolerance
        self.zj = None
        self.fj = None
        self.wj = None
        self.ad_derivatives = []
        self.success = True
        
    @staticmethod
    @jax.jit
    def smooth_barycentric_eval(x, zj, fj, wj, tolerance=1e-8):
        """Smooth version of barycentric evaluation using tanh transition"""
        # Standard barycentric formula
        num = jnp.sum(wj * fj / (x - zj))
        den = jnp.sum(wj / (x - zj))
        val_interp = num / (den + 1e-12)
        
        # Direct value at closest support point
        dists_sq = (x - zj)**2
        idx = jnp.argmin(dists_sq)
        val_direct = fj[idx]
        min_dist_sq = dists_sq[idx]
        
        # Smooth transition using tanh
        # When x is far from any support point: activation ≈ 0, use interpolation
        # When x is close to a support point: activation ≈ 1, use direct value
        activation = 0.5 * (1.0 + jnp.tanh(-min_dist_sq / tolerance + 10))
        
        return activation * val_direct + (1 - activation) * val_interp
        
    def _fit_implementation(self):
        """AAA_FullOpt with smooth barycentric evaluation"""
        best_model = {'bic': np.inf, 'params': None, 'm': 0}
        n_data_points = len(self.t)
        max_possible_m = min(20, n_data_points // 4)
        
        y_scale = jnp.std(self.y)
        lambda_reg = 1e-4 * y_scale if y_scale > 1e-9 else 1e-4
        domain_scale = float(jnp.max(self.t) - jnp.min(self.t))
        
        for m_target in range(3, max_possible_m):
            try:
                aaa_obj = AAA(self.t, self.y, max_terms=m_target)
                zj_initial = jnp.array(aaa_obj.support_points)
                fj_initial = jnp.array(aaa_obj.support_values)
                wj_initial = jnp.array(aaa_obj.weights)
                
                m_actual = len(zj_initial)
                if m_actual <= best_model.get('m', 0):
                    continue
            except Exception:
                continue
                
            def objective_func(params):
                zj, fj, wj = jnp.split(params, 3)
                
                # Use smooth barycentric evaluation
                vmap_bary_eval = jax.vmap(
                    lambda x: self.smooth_barycentric_eval(x, zj, fj, wj, self.smooth_tolerance)
                )
                y_pred = vmap_bary_eval(self.t)
                error_term = jnp.sum((self.y - y_pred)**2)
                
                # Enhanced regularization on finer grid
                t_fine = jnp.linspace(jnp.min(self.t), jnp.max(self.t), 3 * len(self.t))
                d2_func = jax.grad(jax.grad(
                    lambda x: self.smooth_barycentric_eval(x, zj, fj, wj, self.smooth_tolerance)
                ))
                d2_values_fine = jax.vmap(d2_func)(t_fine)
                smoothness_term = jnp.mean(d2_values_fine**2)
                
                # Separation penalty
                zj_sorted = jnp.sort(zj)
                dists = jnp.diff(zj_sorted)
                min_dist_allowed = 1e-4 * domain_scale
                separation_penalty = jnp.sum(jax.nn.relu(min_dist_allowed - dists))
                
                return error_term + lambda_reg * smoothness_term + 1e-2 * separation_penalty
                
            objective_with_grad = jax.jit(jax.value_and_grad(objective_func))
            
            def scipy_objective(params_flat):
                val, grad = objective_with_grad(params_flat)
                if jnp.isnan(val) or jnp.any(jnp.isnan(grad)):
                    return np.inf, np.zeros_like(params_flat)
                return np.array(val, dtype=np.float64), np.array(grad, dtype=np.float64)
                
            initial_params = jnp.concatenate([zj_initial, fj_initial, wj_initial])
            
            try:
                result = minimize(
                    scipy_objective,
                    initial_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if not result.success:
                    continue
                    
                final_params = result.x
                zj_final, fj_final, wj_final = jnp.split(final_params, 3)
                
                # Calculate BIC
                y_pred_final = jax.vmap(
                    lambda x: self.smooth_barycentric_eval(x, zj_final, fj_final, wj_final, self.smooth_tolerance)
                )(self.t)
                pure_rss = jnp.sum((self.y - y_pred_final)**2)
                k = 3 * m_actual
                bic = k * np.log(n_data_points) + n_data_points * np.log(pure_rss / n_data_points + 1e-12)
                
                if bic < best_model['bic']:
                    best_model.update({'bic': bic, 'params': final_params, 'm': m_actual})
                    
            except Exception:
                continue
                
        if best_model['params'] is None:
            self.success = False
            return
            
        self.zj, self.fj, self.wj = jnp.split(best_model['params'], 3)
        
        def single_eval(x):
            return self.smooth_barycentric_eval(x, self.zj, self.fj, self.wj, self.smooth_tolerance)
            
        self.ad_derivatives = [jax.jit(single_eval)]
        for _ in range(self.max_derivative_supported):
            self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))
            
    def _evaluate_function(self, t_eval):
        if not self.success:
            return np.full_like(t_eval, np.nan)
        return np.array(jax.vmap(self.ad_derivatives[0])(t_eval))
        
    def _evaluate_derivative(self, t_eval, order):
        if not self.success or order >= len(self.ad_derivatives):
            return np.full_like(t_eval, np.nan)
        return np.array(jax.vmap(self.ad_derivatives[order])(t_eval))


def test_stable_implementations():
    """Test the stable AAA implementations"""
    
    # Create test data
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 51)
    y_clean = np.sin(2*t) + 0.3*np.cos(3*t)
    
    # Test different noise levels
    noise_levels = [0.01, 0.05, 0.1]
    
    results = []
    
    for noise in noise_levels:
        y_noisy = y_clean + noise * np.random.randn(len(t))
        
        print(f"\nTesting with noise level: {noise}")
        
        # Test two-stage approach
        two_stage = AAA_TwoStage_Approximator(t, y_noisy)
        try:
            two_stage.fit()
            eval_result = two_stage.evaluate(t, max_derivative=2)
            
            rmse_y = np.sqrt(np.mean((eval_result['y'] - y_clean)**2))
            print(f"  TwoStage: Success={two_stage.success}, RMSE={rmse_y:.3e}")
            if hasattr(two_stage, 'stage2_bic') and two_stage.stage2_bic is not None:
                print(f"    Refinement improved BIC: {two_stage.stage1_bic:.1f} -> {two_stage.stage2_bic:.1f}")
            else:
                print(f"    No refinement applied (Stage 1 BIC: {two_stage.stage1_bic:.1f})")
                
        except Exception as e:
            print(f"  TwoStage: FAILED - {e}")
            
        # Test smooth barycentric approach
        smooth_bary = AAA_SmoothBarycentric_Approximator(t, y_noisy)
        try:
            smooth_bary.fit()
            eval_result = smooth_bary.evaluate(t, max_derivative=2)
            
            rmse_y = np.sqrt(np.mean((eval_result['y'] - y_clean)**2))
            print(f"  SmoothBary: Success={smooth_bary.success}, RMSE={rmse_y:.3e}")
                
        except Exception as e:
            print(f"  SmoothBary: FAILED - {e}")


if __name__ == "__main__":
    test_stable_implementations()