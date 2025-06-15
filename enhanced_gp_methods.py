#!/usr/bin/env python3
"""
Enhanced Gaussian Process Methods
Test different kernels and Matérn smoothness parameters
"""

import numpy as np
import jax
import jax.numpy as jnp
import tinygp
from scipy.optimize import minimize
from jax import flatten_util
from comprehensive_methods_library import DerivativeApproximator

# Configure JAX for 64-bit precision
from jax import config
config.update("jax_enable_x64", True)

class EnhancedGPApproximator(DerivativeApproximator):
    """
    Gaussian Process Regression using the tinygp library for JAX-based AD.
    This implementation optimizes the kernel amplitude, length scale, and observation noise.
    """
    def __init__(self, t, y, name="GP_tinygp", kernel_type='rbf_iso'):
        super().__init__(t, y, name)
        self.kernel_type = kernel_type
        self.ad_derivatives = []
        self.gp = None

    def _get_kernel_builder_and_params(self):
        """Returns a function that builds a kernel from parameters, and the initial parameters."""
        if self.kernel_type == 'rbf_iso':
            builder = lambda p: tinygp.kernels.ExpSquared(scale=jnp.exp(p["log_scale"]))
            params = {"log_scale": jnp.log(1.0)}
        elif self.kernel_type == 'matern_1.5':
            builder = lambda p: tinygp.kernels.Matern32(scale=jnp.exp(p["log_scale"]))
            params = {"log_scale": jnp.log(1.0)}
        elif self.kernel_type == 'matern_2.5':
            builder = lambda p: tinygp.kernels.Matern52(scale=jnp.exp(p["log_scale"]))
            params = {"log_scale": jnp.log(1.0)}
        elif self.kernel_type == 'periodic':
            builder = lambda p: tinygp.kernels.ExpSineSquared(
                scale=jnp.exp(p["log_scale"]), gamma=jnp.exp(p["log_gamma"])
            )
            params = {"log_scale": jnp.log(1.0), "log_gamma": jnp.log(1.0)}
        else:
            return None, None
        return builder, params

    def _fit_implementation(self):
        base_kernel_builder, kernel_params = self._get_kernel_builder_and_params()
        if base_kernel_builder is None:
            self.gp = None
            return

        initial_params = {
            "log_amp": jnp.log(np.std(self.y)),
            "log_diag": jnp.log(0.1 * np.std(self.y)),
            "kernel_params": kernel_params,
        }

        @jax.jit
        def loss(params):
            kernel = jnp.exp(params["log_amp"]) * base_kernel_builder(params["kernel_params"])
            gp = tinygp.GaussianProcess(kernel, self.t, diag=jnp.exp(params["log_diag"]))
            return -gp.log_probability(self.y)

        loss_and_grad = jax.jit(jax.value_and_grad(loss))
        
        x0_flat, unflatten = flatten_util.ravel_pytree(initial_params)

        def objective_scipy(x_flat):
            params = unflatten(x_flat)
            val, grad_pytree = loss_and_grad(params)
            grad_flat, _ = flatten_util.ravel_pytree(grad_pytree)
            # Ensure output is float64 for scipy
            return np.array(val, dtype=np.float64), np.array(grad_flat, dtype=np.float64)

        try:
            solution = minimize(objective_scipy, x0_flat, jac=True, method="L-BFGS-B")
            if not solution.success:
                raise RuntimeError(f"Optimizer failed: {solution.message}")

            final_params = unflatten(solution.x)
            final_kernel = jnp.exp(final_params["log_amp"]) * base_kernel_builder(final_params["kernel_params"])
            self.gp = tinygp.GaussianProcess(final_kernel, self.t, diag=jnp.exp(final_params["log_diag"]))

            def predict_mean(t_point):
                _, conditioned_gp = self.gp.condition(self.y, jnp.atleast_1d(t_point))
                return conditioned_gp.mean[0]

            self.ad_derivatives = [jax.jit(predict_mean)]
            for _ in range(5):
                self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))
        except Exception as e:
            self.gp = None
            if not hasattr(self, '_warned_fit_fail'):
                print(f"WARNING: GP fit failed for {self.name}. Reason: {str(e)[:100]}")
                self._warned_fit_fail = True

    def _evaluate_function(self, t_eval):
        if self.gp is None:
            return np.full_like(t_eval, np.nan)
        _, conditioned_gp = self.gp.condition(self.y, t_eval)
        return np.array(conditioned_gp.mean)

    def _evaluate_derivative(self, t_eval, order):
        if self.gp is None or not self.ad_derivatives:
            if not hasattr(self, '_warned_gp_deriv'):
                 print(f"WARNING: Derivatives for {self.name} could not be computed (unsupported kernel or fit failed). Returning NaNs.")
                 self._warned_gp_deriv = True
            return np.full_like(t_eval, np.nan)

        if order < len(self.ad_derivatives):
            vmap_grad = jax.vmap(self.ad_derivatives[order])
            return np.array(vmap_grad(jnp.array(t_eval)))
        else:
            return np.zeros_like(t_eval)

def create_enhanced_gp_methods(t, y):
    """Create instances of all available tinygp-based methods"""
    methods = {}
    
    # Supported kernel types in this refactoring
    kernel_map = {
        'GP_RBF_Iso': 'rbf_iso',
        'GP_Matern_1.5': 'matern_1.5',
        'GP_Matern_2.5': 'matern_2.5',
        'GP_Periodic': 'periodic'
    }

    for name, kernel_type in kernel_map.items():
        methods[name] = EnhancedGPApproximator(t, y, name, kernel_type=kernel_type)

    # Note: Matern 0.5, 5.0 and Rational Quadratic from the original script are not 
    # included, as they do not have a direct, simple equivalent in tinygp's base 
    # kernels or would require more complex construction.
    
    return methods

if __name__ == "__main__":
    # Test the enhanced GP methods
    t = np.linspace(0, 2*np.pi, 50)
    y = np.sin(t) + 0.01 * np.random.randn(len(t))
    
    methods = create_enhanced_gp_methods(t, y)
    
    print(f"Created {len(methods)} enhanced GP methods:")
    for name in methods:
        print(f"  - {name}")
    
    # Quick test
    t_eval = np.linspace(0, 2*np.pi, 20)
    
    print("\\nTesting methods:")
    for name, method in methods.items():
        try:
            result = method.evaluate(t_eval, max_derivative=2)
            success = "✓" if result['success'] else "❌"
            print(f"  {success} {name}")
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")