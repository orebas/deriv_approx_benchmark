#!/usr/bin/env python3
"""
Comprehensive Methods Library for Derivative Approximation
Uses existing packages to implement a wide variety of methods
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate, optimize, sparse
from scipy.special import eval_chebyt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings
import jax
import jax.numpy as jnp
import tinygp
from scipy.optimize import minimize
from jax import flatten_util
from scipy.interpolate import AAA

# Configure JAX to use 64-bit precision to match numpy
from jax import config
config.update("jax_enable_x64", True)

warnings.filterwarnings('ignore')

class DerivativeApproximator:
    """Base class for all derivative approximation methods"""
    
    def __init__(self, t, y, name="Unknown"):
        self.t = t
        self.y = y
        self.name = name
        self.fitted = False
        self.fit_time = 0
    
    def fit(self):
        """Fit the method to the data"""
        import time
        start_time = time.time()
        self._fit_implementation()
        self.fit_time = time.time() - start_time
        self.fitted = True
    
    def _fit_implementation(self):
        """Override this in subclasses"""
        raise NotImplementedError
    
    def evaluate(self, t_eval, max_derivative=5):
        """Evaluate function and derivatives"""
        if not self.fitted:
            self.fit()
        
        results = {}
        results['y'] = self._evaluate_function(t_eval)
        for d in range(1, max_derivative + 1):
            results[f'd{d}'] = self._evaluate_derivative(t_eval, d)
        results['success'] = True
        
        return results
    
    def _evaluate_function(self, t_eval):
        raise NotImplementedError
    
    def _evaluate_derivative(self, t_eval, order):
        raise NotImplementedError

# =============================================================================
# INTERPOLATION-BASED METHODS
# =============================================================================

class CubicSplineApproximator(DerivativeApproximator):
    """Cubic spline interpolation (scipy)"""
    
    def _fit_implementation(self):
        self.spline = interpolate.CubicSpline(self.t, self.y)
    
    def _evaluate_function(self, t_eval):
        return self.spline(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        return self.spline.derivative(order)(t_eval)

class UnivariateSplineApproximator(DerivativeApproximator):
    """Univariate spline with smoothing (scipy)"""
    
    def __init__(self, t, y, name="UnivariateSpline", s=None, k=3):
        super().__init__(t, y, name)
        self.s = s  # Smoothing factor
        self.k = k  # Spline degree
    
    def _fit_implementation(self):
        self.spline = interpolate.UnivariateSpline(self.t, self.y, s=self.s, k=self.k)
    
    def _evaluate_function(self, t_eval):
        return self.spline(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        return self.spline.derivative(order)(t_eval)

class RBFInterpolatorApproximator(DerivativeApproximator):
    """Radial Basis Function interpolation (scipy)"""
    
    def __init__(self, t, y, name="RBF", kernel='thin_plate_spline', epsilon=None):
        super().__init__(t, y, name)
        self.kernel = kernel
        self.epsilon = epsilon
        self.max_derivative_supported = 0 # Derivatives are not analytically supported

    def _fit_implementation(self):
        # Reshape for scipy RBFInterpolator
        points = self.t.reshape(-1, 1)
        # Pass epsilon only if it's not None
        if self.epsilon is not None:
            self.rbf = interpolate.RBFInterpolator(points, self.y, kernel=self.kernel, epsilon=self.epsilon)
        else:
            self.rbf = interpolate.RBFInterpolator(points, self.y, kernel=self.kernel)
    
    def _evaluate_function(self, t_eval):
        return self.rbf(t_eval.reshape(-1, 1))

    def _evaluate_derivatives(self, t_eval, max_derivative):
        # RBFInterpolator doesn't have a public derivative method
        print(f"WARNING: Analytical derivatives for {self.name} are not implemented. Returning NaNs.")
        derivs = {}
        for i in range(1, max_derivative + 1):
            derivs[f'd{i}'] = np.full_like(t_eval, np.nan)
        return derivs

# =============================================================================
# GAUSSIAN PROCESS METHODS
# =============================================================================

class GPRegressionApproximator(DerivativeApproximator):
    """Gaussian Process Regression using tinygp for AD"""
    
    def __init__(self, t, y, name="GP_RBF", kernel_type='rbf'):
        super().__init__(t, y, name)
        self.kernel_type = kernel_type
        self.ad_derivatives = []
        self.gp = None

    def _get_kernel_builder_and_params(self):
        """Returns a kernel builder and initial parameters."""
        if self.kernel_type == 'rbf':
            builder = lambda p: tinygp.kernels.ExpSquared(scale=jnp.exp(p["log_scale"]))
        elif self.kernel_type == 'matern':
            builder = lambda p: tinygp.kernels.Matern32(scale=jnp.exp(p["log_scale"]))
        else:
            raise ValueError(f"Unsupported kernel type for GPRegressionApproximator: {self.kernel_type}")
        
        initial_params = {"log_scale": jnp.log(1.0)}
        return builder, initial_params

    def _fit_implementation(self):
        base_kernel_builder, kernel_params = self._get_kernel_builder_and_params()

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
            return np.full_like(t_eval, np.nan)
        if order < len(self.ad_derivatives):
            vmap_grad = jax.vmap(self.ad_derivatives[order])
            return np.array(vmap_grad(jnp.array(t_eval)))
        else:
            return np.zeros_like(t_eval)

# =============================================================================
# POLYNOMIAL METHODS
# =============================================================================

class ChebyshevApproximator(DerivativeApproximator):
    """Chebyshev polynomial approximation (scipy/numpy)"""
    
    def __init__(self, t, y, name="Chebyshev", degree=None):
        super().__init__(t, y, name)
        self.degree = min(degree or len(y)-1, 20)  # Reasonable upper limit
    
    def _fit_implementation(self):
        # Map to [-1, 1]
        self.t_min, self.t_max = self.t.min(), self.t.max()
        t_cheb = 2 * (self.t - self.t_min) / (self.t_max - self.t_min) - 1
        
        # Fit Chebyshev polynomial
        self.coeffs = np.polynomial.chebyshev.chebfit(t_cheb, self.y, self.degree)
    
    def _map_to_cheb(self, t_eval):
        return 2 * (t_eval - self.t_min) / (self.t_max - self.t_min) - 1
    
    def _evaluate_function(self, t_eval):
        t_cheb = self._map_to_cheb(t_eval)
        return np.polynomial.chebyshev.chebval(t_cheb, self.coeffs)
    
    def _evaluate_derivative(self, t_eval, order):
        # Derivative of Chebyshev polynomials
        deriv_coeffs = self.coeffs.copy()
        domain_factor = (2 / (self.t_max - self.t_min)) ** order
        
        for _ in range(order):
            deriv_coeffs = np.polynomial.chebyshev.chebder(deriv_coeffs)
        
        t_cheb = self._map_to_cheb(t_eval)
        return domain_factor * np.polynomial.chebyshev.chebval(t_cheb, deriv_coeffs)

class PolynomialRegressionApproximator(DerivativeApproximator):
    """Polynomial regression (sklearn)"""
    
    def __init__(self, t, y, name="Polynomial", degree=5):
        super().__init__(t, y, name)
        self.degree = degree
        self.ad_derivatives = []
    
    def _fit_implementation(self):
        from sklearn.pipeline import Pipeline
        
        # Use include_bias=False because Ridge has its own intercept.
        # This makes getting the final coefficients cleaner.
        self.poly_reg = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('ridge', Ridge(alpha=1.0, fit_intercept=True))
        ])
        
        X = self.t.reshape(-1, 1)
        self.poly_reg.fit(X, self.y)

        # Extract coefficients for AD
        ridge_model = self.poly_reg.named_steps['ridge']
        self.coeffs = jnp.array(ridge_model.coef_)
        self.intercept = jnp.array(ridge_model.intercept_)

        # Define the polynomial function for JAX AD
        def poly_func(t):
            # Create terms t, t^2, t^3, ...
            powers = jnp.power(t, jnp.arange(1, self.degree + 1))
            return jnp.dot(powers, self.coeffs) + self.intercept

        # Create jitted derivative functions up to 5th order
        self.ad_derivatives = [jax.jit(poly_func)]
        for i in range(5): # Max 5 derivatives
            self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))
    
    def _evaluate_function(self, t_eval):
        X_eval = t_eval.reshape(-1, 1)
        return self.poly_reg.predict(X_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        if order < len(self.ad_derivatives):
            # Vectorize the jitted function for the given order (func is at 0, d1 is at 1, etc)
            vmap_grad = jax.vmap(self.ad_derivatives[order])
            return np.array(vmap_grad(jnp.array(t_eval)))
        else:
            # Fallback for very high orders
            return np.zeros_like(t_eval)

# =============================================================================
# SMOOTHING FILTERS
# =============================================================================

class SavitzkyGolayApproximator(DerivativeApproximator):
    """Savitzky-Golay filter (scipy)"""
    
    def __init__(self, t, y, name="SavGol", window_length=None, polyorder=3):
        super().__init__(t, y, name)
        
        if window_length is None:
            window_length = min(len(y) // 3, 21)
            if window_length % 2 == 0:
                window_length += 1
        
        self.window_length = max(polyorder + 1, window_length)
        self.polyorder = min(polyorder, self.window_length - 1)
    
    def _fit_implementation(self):
        # Apply Savitzky-Golay filter
        if len(self.y) >= self.window_length:
            self.y_smooth = signal.savgol_filter(self.y, self.window_length, self.polyorder)
        else:
            self.y_smooth = self.y.copy()
        
        # Create spline for interpolation
        self.spline = interpolate.CubicSpline(self.t, self.y_smooth)
    
    def _evaluate_function(self, t_eval):
        return self.spline(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        return self.spline.derivative(order)(t_eval)

class ButterFilterApproximator(DerivativeApproximator):
    """Butterworth filter followed by spline (scipy)"""
    
    def __init__(self, t, y, name="Butterworth", cutoff_freq=0.1, order=4):
        super().__init__(t, y, name)
        self.cutoff_freq = cutoff_freq
        self.filter_order = order
    
    def _fit_implementation(self):
        # Design Butterworth filter
        dt = np.mean(np.diff(self.t))
        nyquist = 0.5 / dt
        normalized_cutoff = self.cutoff_freq / nyquist
        
        if normalized_cutoff < 1.0:
            b, a = signal.butter(self.filter_order, normalized_cutoff)
            # Apply filter
            self.y_filtered = signal.filtfilt(b, a, self.y)
        else:
            self.y_filtered = self.y.copy()
        
        # Create spline
        self.spline = interpolate.CubicSpline(self.t, self.y_filtered)
    
    def _evaluate_function(self, t_eval):
        return self.spline(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        return self.spline.derivative(order)(t_eval)

# =============================================================================
# MACHINE LEARNING METHODS
# =============================================================================

class RandomForestApproximator(DerivativeApproximator):
    """Random Forest regression (sklearn)"""
    
    def __init__(self, t, y, name="RandomForest", n_estimators=100):
        super().__init__(t, y, name)
        self.n_estimators = n_estimators
    
    def _fit_implementation(self):
        self.rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
        X = self.t.reshape(-1, 1)
        self.rf.fit(X, self.y)
    
    def _evaluate_function(self, t_eval):
        X_eval = t_eval.reshape(-1, 1)
        return self.rf.predict(X_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        # NOTE: Random Forest models produce piecewise-constant prediction surfaces,
        # so their derivatives are zero or undefined. This method is fundamentally
        # unsuitable for derivative approximation via AD or finite differences.
        if not hasattr(self, '_warned_rf'):
            print(f"WARNING: {self.name} is not suitable for derivative calculation. Returning NaNs.")
            self._warned_rf = True
        return np.full_like(t_eval, np.nan)

class SVRApproximator(DerivativeApproximator):
    """Support Vector Regression (sklearn)"""
    
    def __init__(self, t, y, name="SVR", kernel='rbf', C=1.0):
        super().__init__(t, y, name)
        self.kernel = kernel
        self.C = C
        self.ad_derivatives = []
    
    def _fit_implementation(self):
        self.svr = SVR(kernel=self.kernel, C=self.C, gamma='scale')
        X = self.t.reshape(-1, 1)
        self.svr.fit(X, self.y)

        # If kernel is not 'rbf', we can't use our analytical AD function.
        if self.kernel != 'rbf':
            self.ad_func = None
            return

        # Extract SVR parameters for JAX
        support_vectors = jnp.array(self.svr.support_vectors_[:, 0])
        dual_coef = jnp.array(self.svr.dual_coef_[0, :])
        intercept = jnp.array(self.svr.intercept_[0])
        gamma = self.svr._gamma

        # Define the SVR prediction function for JAX AD
        def svr_func(t):
            # RBF kernel: exp(-gamma * ||x - y||^2)
            kernel_vals = jnp.exp(-gamma * (support_vectors - t)**2)
            return jnp.dot(dual_coef, kernel_vals) + intercept
        
        # Create jitted derivative functions
        self.ad_derivatives = [jax.jit(svr_func)]
        for i in range(5): # Max 5 derivatives
            self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))

    def _evaluate_function(self, t_eval):
        X_eval = t_eval.reshape(-1, 1)
        return self.svr.predict(X_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        if self.kernel != 'rbf' or not self.ad_derivatives:
            if not hasattr(self, '_warned_svr'):
                print(f"WARNING: Derivatives for SVR with kernel '{self.kernel}' not implemented. Returning NaNs.")
                self._warned_svr = True
            return np.full_like(t_eval, np.nan)

        if order < len(self.ad_derivatives):
            vmap_grad = jax.vmap(self.ad_derivatives[order])
            return np.array(vmap_grad(jnp.array(t_eval)))
        else:
            return np.zeros_like(t_eval)

# =============================================================================
# SPECTRAL METHODS
# =============================================================================

class FourierApproximator(DerivativeApproximator):
    """Fourier series approximation (scipy/numpy)"""
    
    def __init__(self, t, y, name="Fourier", n_harmonics=None):
        super().__init__(t, y, name)
        self.n_harmonics = n_harmonics or min(len(y) // 4, 50)
    
    def _fit_implementation(self):
        n = len(self.y)
        
        # Ensure periodic extension if needed
        self.period = self.t[-1] - self.t[0]
        self.t0 = self.t[0]
        
        # FFT
        fft_y = np.fft.fft(self.y)
        self.freqs = np.fft.fftfreq(n, (self.t[1] - self.t[0]))
        
        # Keep only the requested harmonics
        self.coeffs = fft_y[:min(self.n_harmonics, n//2)]
        self.freqs_used = self.freqs[:min(self.n_harmonics, n//2)]
    
    def _evaluate_function(self, t_eval):
        result = np.real(self.coeffs[0]) / len(self.y) * np.ones_like(t_eval)
        
        for k in range(1, len(self.coeffs)):
            if k < len(self.coeffs) // 2:
                phase = 2j * np.pi * self.freqs_used[k] * (t_eval - self.t0)
                result += 2 * np.real(self.coeffs[k] * np.exp(phase)) / len(self.y)
        
        return result
    
    def _evaluate_derivative(self, t_eval, order):
        result = np.zeros_like(t_eval)
        
        for k in range(1, len(self.coeffs)):
            if k < len(self.coeffs) // 2:
                phase = 2j * np.pi * self.freqs_used[k] * (t_eval - self.t0)
                deriv_factor = (2j * np.pi * self.freqs_used[k]) ** order
                result += 2 * np.real(self.coeffs[k] * deriv_factor * np.exp(phase)) / len(self.y)
        
        return result

# =============================================================================
# FINITE DIFFERENCE METHODS
# =============================================================================

class FiniteDifferenceApproximator(DerivativeApproximator):
    """Higher-order finite differences (numpy)"""
    
    def __init__(self, t, y, name="FiniteDiff", order=5):
        super().__init__(t, y, name)
        self.fd_order = order
    
    def _fit_implementation(self):
        # Use spline for smooth interpolation
        self.spline = interpolate.CubicSpline(self.t, self.y)
    
    def _evaluate_function(self, t_eval):
        return self.spline(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        # Use analytical derivatives from spline (more accurate than finite differences)
        return self.spline.derivative(order)(t_eval)

# =============================================================================
#  New Approximator: AAA with Least-Squares Refinement
# =============================================================================

@jax.jit
def barycentric_eval(x, zj, fj, wj):
    """
    JAX-based evaluation of a barycentric rational function.
    Handles evaluation at support points correctly for AD.
    """
    # Find if x is one of the support points
    is_support_point = jnp.any(jnp.isclose(x, zj))
    
    # If it is a support point, find its index and return the corresponding value
    # If not, compute the standard barycentric formula
    def true_fn():
        idx = jnp.argmin(jnp.abs(x - zj))
        return fj[idx]

    def false_fn():
        num = jnp.sum(wj * fj / (x - zj))
        den = jnp.sum(wj / (x - zj))
        # Add a small epsilon to the denominator to avoid division by zero
        # in cases where the denominator might be close to zero for non-support points.
        return num / (den + 1e-12)

    return jax.lax.cond(is_support_point, true_fn, false_fn)


class AAALeastSquaresApproximator(DerivativeApproximator):
    """
    Approximates a function using AAA to find support points, then
    refines the weights and values using least-squares optimization.
    Derivatives are computed using JAX for accuracy.
    """
    def __init__(self, t, y, name="AAA_LS"):
        super().__init__(t, y, name)
        self.max_derivative_supported = 5
        self.zj = None
        self.fj = None
        self.wj = None
        self.ad_derivatives = []
        self.success = True

    def _fit_implementation(self):
        
        best_model = {'bic': np.inf, 'params': None, 'zj': None, 'm': 0}
        n_data_points = len(self.t)
        max_possible_m = min(25, n_data_points // 3) 
        
        # Make regularization adaptive to data scale
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
                if not result.success: continue
                
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
            self.success = False
            if not hasattr(self, '_warned_aaa_fit'):
                print(f"\nWARNING: AAA_LS failed to find any stable model for {self.name}.")
                self._warned_aaa_fit = True
            return

        self.zj = best_model['zj']
        self.fj, self.wj = jnp.split(best_model['params'], 2)
        
        def single_eval(x):
            return barycentric_eval(x, self.zj, self.fj, self.wj)
            
        self.ad_derivatives = [jax.jit(single_eval)]
        for _ in range(self.max_derivative_supported):
            self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))

    def _evaluate_function(self, t_eval):
        if not self.success: return np.full_like(t_eval, np.nan)
        # Vmap the 0-th order function for evaluation on an array
        return np.array(jax.vmap(self.ad_derivatives[0])(t_eval))

    def _evaluate_derivative(self, t_eval, order):
        if not self.success or order >= len(self.ad_derivatives):
            return np.full_like(t_eval, np.nan)
        # Vmap the n-th order derivative function for evaluation
        return np.array(jax.vmap(self.ad_derivatives[order])(t_eval))

# =============================================================================
#  New Approximator: AAA with Full Optimization (Level 2)
# =============================================================================

class AAA_FullOpt_Approximator(DerivativeApproximator):
    """
    Approximates a function by using AAA for an initial guess, then fully
    optimizing all parameters (support points, values, and weights)
    using a regularized least-squares objective.
    """
    def __init__(self, t, y, name="AAA_FullOpt"):
        super().__init__(t, y, name)
        self.max_derivative_supported = 5
        self.zj = None
        self.fj = None
        self.wj = None
        self.ad_derivatives = []
        self.success = True

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
            except Exception as e:
                continue

            def objective_func(params):
                zj, fj, wj = jnp.split(params, 3)
                
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

            initial_params = jnp.concatenate([zj_initial, fj_initial, wj_initial])
            
            try:
                result = minimize(
                    scipy_objective,
                    initial_params,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if not result.success: continue
                
                final_params = result.x
                
                # Re-evaluate the error term (RSS) without the penalty terms
                # for a correct BIC calculation.
                zj_final, fj_final, wj_final = jnp.split(final_params, 3)
                y_pred_final = jax.vmap(lambda x: barycentric_eval(x, zj_final, fj_final, wj_final))(self.t)
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
            return barycentric_eval(x, self.zj, self.fj, self.wj)
            
        self.ad_derivatives = [jax.jit(single_eval)]
        for _ in range(self.max_derivative_supported):
            self.ad_derivatives.append(jax.jit(jax.grad(self.ad_derivatives[-1])))

    def _evaluate_function(self, t_eval):
        if not self.success: return np.full_like(t_eval, np.nan)
        # Vmap the 0-th order function for evaluation on an array
        return np.array(jax.vmap(self.ad_derivatives[0])(t_eval))

    def _evaluate_derivative(self, t_eval, order):
        if not self.success or order >= len(self.ad_derivatives):
            return np.full_like(t_eval, np.nan)
        # Vmap the n-th order derivative function for evaluation
        return np.array(jax.vmap(self.ad_derivatives[order])(t_eval))

# =============================================================================
#  KalmanGrad Approximator
# =============================================================================

class KalmanGradApproximator(DerivativeApproximator):
    """
    Approximates derivatives using Bayesian filtering/smoothing via KalmanGrad.
    Uses Kalman filtering to estimate function and derivatives from noisy data.
    """
    def __init__(self, t, y, name="KalmanGrad"):
        super().__init__(t, y, name)
        self.max_derivative_supported = 5
        self.smoother_states = None
        self.filter_times = None
        self.success = True
        self.obs_noise_std = 0.01
        self.final_cov = 0.0001
        
    def _fit_implementation(self):
        try:
            from kalmangrad import grad
            from scipy.interpolate import interp1d
            
            # Estimate noise level from data if not specified
            if len(self.y) > 10:
                # Use median absolute deviation to estimate noise
                diff_y = np.diff(self.y)
                self.obs_noise_std = max(1.48 * np.median(np.abs(diff_y - np.median(diff_y))), 1e-6)
            
            # Adjust final_cov based on data scale
            data_scale = np.std(self.y)
            self.final_cov = min(0.01 * data_scale**2, 1.0)
            
            # Run KalmanGrad with automatic parameter estimation
            self.smoother_states, self.filter_times = grad(
                self.y, self.t, 
                n=self.max_derivative_supported,
                obs_noise_std=self.obs_noise_std,
                online=False,  # Use offline smoothing for better accuracy
                final_cov=self.final_cov
            )
            
            self.success = True
            
        except Exception as e:
            print(f"KalmanGrad fitting failed: {e}")
            self.success = False
    
    def _evaluate_function(self, t_eval):
        if not self.success or self.smoother_states is None:
            return np.full_like(t_eval, np.nan)
        
        try:
            from scipy.interpolate import interp1d
            
            # Extract function values from smoother states
            # Note: mean is a method, not an attribute
            y_kg = np.array([state.mean()[0] for state in self.smoother_states])
            
            # Convert times to array if it's a list
            times_array = np.array(self.filter_times) if isinstance(self.filter_times, list) else self.filter_times
            
            # Interpolate to evaluation points
            interp_func = interp1d(times_array, y_kg, 
                                 kind='cubic', bounds_error=False, 
                                 fill_value='extrapolate')
            return interp_func(t_eval)
            
        except Exception:
            return np.full_like(t_eval, np.nan)
    
    def _evaluate_derivative(self, t_eval, order):
        if not self.success or self.smoother_states is None:
            return np.full_like(t_eval, np.nan)
        
        if order == 0:
            return self._evaluate_function(t_eval)
        
        if order > self.max_derivative_supported:
            return np.full_like(t_eval, np.nan)
        
        try:
            from scipy.interpolate import interp1d
            
            # Extract derivative values from smoother states
            # Note: mean is a method, not an attribute
            dy_kg = np.array([state.mean()[order] for state in self.smoother_states])
            
            # Convert times to array if it's a list
            times_array = np.array(self.filter_times) if isinstance(self.filter_times, list) else self.filter_times
            
            # Interpolate to evaluation points
            interp_func = interp1d(times_array, dy_kg, 
                                 kind='cubic', bounds_error=False, 
                                 fill_value='extrapolate')
            return interp_func(t_eval)
            
        except Exception:
            return np.full_like(t_eval, np.nan)

# =============================================================================
# METHOD FACTORY
# =============================================================================

def create_all_methods(t, y):
    """Create instances of all available methods"""
    
    methods = {}
    
    # Interpolation methods
    methods['CubicSpline'] = CubicSplineApproximator(t, y, "CubicSpline")
    methods['SmoothingSpline'] = UnivariateSplineApproximator(t, y, "SmoothingSpline", s=0.1)
    methods['RBF_ThinPlate'] = RBFInterpolatorApproximator(t, y, "RBF_ThinPlate", 'thin_plate_spline')
    methods['RBF_Multiquadric'] = RBFInterpolatorApproximator(t, y, "RBF_Multiquadric", 'multiquadric', epsilon=1.0)
    
    # Gaussian Process methods
    methods['GP_RBF'] = GPRegressionApproximator(t, y, "GP_RBF", 'rbf')
    methods['GP_Matern'] = GPRegressionApproximator(t, y, "GP_Matern", 'matern')
    
    # Polynomial methods
    methods['Chebyshev'] = ChebyshevApproximator(t, y, "Chebyshev", degree=min(15, len(y)-1))
    methods['Polynomial'] = PolynomialRegressionApproximator(t, y, "Polynomial", degree=min(8, len(y)-1))
    
    # Smoothing filters
    methods['SavitzkyGolay'] = SavitzkyGolayApproximator(t, y, "SavitzkyGolay")
    methods['Butterworth'] = ButterFilterApproximator(t, y, "Butterworth")
    
    # Machine learning methods
    methods['RandomForest'] = RandomForestApproximator(t, y, "RandomForest")
    methods['SVR'] = SVRApproximator(t, y, "SVR")
    
    # Spectral methods
    methods['Fourier'] = FourierApproximator(t, y, "Fourier")
    
    # Finite difference methods
    methods['FiniteDiff'] = FiniteDifferenceApproximator(t, y, "FiniteDiff")
    
    # Advanced approximators
    methods['AAA_LS'] = AAALeastSquaresApproximator(t, y)
    methods['AAA_FullOpt'] = AAA_FullOpt_Approximator(t, y)
    methods['KalmanGrad'] = KalmanGradApproximator(t, y, "KalmanGrad")
    
    return methods

def get_base_method_names():
    """Return the names of the base methods available."""
    # Create dummy arrays for instantiation
    t = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    return list(create_all_methods(t, y).keys())

def get_method_categories():
    """Return categorization of methods"""
    return {
        'Interpolation': ['CubicSpline', 'SmoothingSpline', 'RBF_ThinPlate', 'RBF_Multiquadric'],
        'Gaussian_Process': ['GP_RBF', 'GP_Matern'],
        'Polynomial': ['Chebyshev', 'Polynomial'],
        'Smoothing': ['SavitzkyGolay', 'Butterworth'],
        'Machine_Learning': ['RandomForest', 'SVR'],
        'Spectral': ['Fourier'],
        'Finite_Difference': ['FiniteDiff'],
        'Advanced': ['AAA_LS', 'AAA_FullOpt', 'KalmanGrad']
    }

if __name__ == "__main__":
    # Example usage
    t = np.linspace(0, 4*np.pi, 50)
    y = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    methods = create_all_methods(t, y)
    
    print(f"Created {len(methods)} methods:")
    for name, method in methods.items():
        print(f"  - {name}")
    
    # Test one method
    t_eval = np.linspace(0, 4*np.pi, 100)
    test_method = methods['SavitzkyGolay']
    result = test_method.evaluate(t_eval)
    
    print(f"\nTest successful: {result['success']}")
    print(f"Function values shape: {result['y'].shape}")
    print(f"First derivative shape: {result['d1'].shape}")