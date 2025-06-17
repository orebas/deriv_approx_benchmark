#!/usr/bin/env python3
"""
Standalone Python implementation of additional derivative approximation methods
Can be called from Julia or run independently
"""

import numpy as np
import pandas as pd
from scipy import signal, interpolate, fft
from scipy.optimize import minimize
import argparse
import json

class DerivativeApproximator:
    """Base class for derivative approximation methods"""
    
    def __init__(self, t, y):
        self.t = np.array(t)
        self.y = np.array(y)
        self.fitted = False
    
    def fit(self):
        """Fit the method to the data"""
        raise NotImplementedError
    
    def evaluate(self, t_eval, max_derivative=5):
        """Evaluate function and derivatives at t_eval points"""
        if not self.fitted:
            self.fit()
        
        results = {}
        results['y'] = self._evaluate_function(t_eval)
        
        for d in range(1, max_derivative + 1):
            results[f'd{d}'] = self._evaluate_derivative(t_eval, d)
        
        return results
    
    def _evaluate_function(self, t_eval):
        """Evaluate function at t_eval"""
        raise NotImplementedError
    
    def _evaluate_derivative(self, t_eval, order):
        """Evaluate derivative of given order at t_eval"""
        raise NotImplementedError

class SavitzkyGolayApproximator(DerivativeApproximator):
    """Savitzky-Golay filter for derivative approximation"""
    
    def __init__(self, t, y, window_length=None, polyorder=3):
        super().__init__(t, y)
        
        # Auto-select window length if not provided
        if window_length is None:
            window_length = min(11, len(y) // 3)
            if window_length % 2 == 0:
                window_length += 1  # Must be odd
        
        self.window_length = max(polyorder + 1, window_length)
        self.polyorder = min(polyorder, self.window_length - 1)
        
    def fit(self):
        """Fit Savitzky-Golay filter"""
        # Apply S-G filter to smooth the data
        self.y_smooth = signal.savgol_filter(self.y, self.window_length, self.polyorder)
        
        # Create spline interpolator for smooth evaluation
        self.spline = interpolate.CubicSpline(self.t, self.y_smooth)
        self.fitted = True
    
    def _evaluate_function(self, t_eval):
        return self.spline(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        return self.spline.derivative(order)(t_eval)

class FourierApproximator(DerivativeApproximator):
    """Fourier-based approximation for periodic functions"""
    
    def __init__(self, t, y, n_harmonics=None):
        super().__init__(t, y)
        
        # Auto-select number of harmonics
        if n_harmonics is None:
            n_harmonics = min(len(y) // 4, 20)
        
        self.n_harmonics = n_harmonics
        
    def fit(self):
        """Fit Fourier series"""
        n = len(self.y)
        
        # Compute FFT
        fft_y = fft.fft(self.y)
        
        # Period and frequencies
        self.period = self.t[-1] - self.t[0]
        self.t0 = self.t[0]
        
        # Store Fourier coefficients
        self.coeffs = fft_y[:self.n_harmonics] / n
        self.freqs = fft.fftfreq(n, (self.t[1] - self.t[0]))[:self.n_harmonics]
        
        self.fitted = True
    
    def _evaluate_function(self, t_eval):
        """Evaluate Fourier series"""
        result = np.real(self.coeffs[0]) * np.ones_like(t_eval)  # DC component
        
        for k in range(1, len(self.coeffs)):
            phase = 2j * np.pi * self.freqs[k] * (t_eval - self.t0)
            if k < len(self.coeffs) // 2:
                # Positive frequencies
                result += 2 * np.real(self.coeffs[k] * np.exp(phase))
            
        return result
    
    def _evaluate_derivative(self, t_eval, order):
        """Evaluate derivative of Fourier series"""
        result = np.zeros_like(t_eval)
        
        for k in range(1, len(self.coeffs)):
            phase = 2j * np.pi * self.freqs[k] * (t_eval - self.t0)
            deriv_factor = (2j * np.pi * self.freqs[k]) ** order
            
            if k < len(self.coeffs) // 2:
                result += 2 * np.real(self.coeffs[k] * deriv_factor * np.exp(phase))
        
        return result

class ChebyshevApproximator(DerivativeApproximator):
    """Chebyshev polynomial approximation"""
    
    def __init__(self, t, y, degree=None):
        super().__init__(t, y)
        
        # Auto-select degree
        if degree is None:
            degree = min(len(y) - 1, 15)
        
        self.degree = degree
    
    def fit(self):
        """Fit Chebyshev polynomial"""
        # Map t to [-1, 1] interval
        self.t_min, self.t_max = self.t.min(), self.t.max()
        t_cheb = 2 * (self.t - self.t_min) / (self.t_max - self.t_min) - 1
        
        # Fit Chebyshev polynomial
        self.cheb_coeffs = np.polynomial.chebyshev.chebfit(t_cheb, self.y, self.degree)
        
        self.fitted = True
    
    def _map_to_cheb_domain(self, t_eval):
        """Map evaluation points to Chebyshev domain [-1, 1]"""
        return 2 * (t_eval - self.t_min) / (self.t_max - self.t_min) - 1
    
    def _evaluate_function(self, t_eval):
        t_cheb = self._map_to_cheb_domain(t_eval)
        return np.polynomial.chebyshev.chebval(t_cheb, self.cheb_coeffs)
    
    def _evaluate_derivative(self, t_eval, order):
        # Compute derivative coefficients
        deriv_coeffs = self.cheb_coeffs.copy()
        
        # Chain rule factor for domain mapping
        domain_factor = (2 / (self.t_max - self.t_min)) ** order
        
        for _ in range(order):
            deriv_coeffs = np.polynomial.chebyshev.chebder(deriv_coeffs)
        
        t_cheb = self._map_to_cheb_domain(t_eval)
        return domain_factor * np.polynomial.chebyshev.chebval(t_cheb, deriv_coeffs)

class FiniteDifferenceApproximator(DerivativeApproximator):
    """Higher-order finite difference approximation"""
    
    def __init__(self, t, y, order=5):
        super().__init__(t, y)
        self.fd_order = order  # Order of finite difference stencil
    
    def fit(self):
        """Create spline interpolator for smooth evaluation"""
        # Use cubic spline for interpolation
        self.spline = interpolate.CubicSpline(self.t, self.y)
        self.fitted = True
    
    def _evaluate_function(self, t_eval):
        return self.spline(t_eval)
    
    def _evaluate_derivative(self, t_eval, order):
        """Use spline derivatives (more robust than finite differences)"""
        return self.spline.derivative(order)(t_eval)

def run_method_comparison(t, y, t_eval, methods=['savgol', 'fourier', 'chebyshev', 'finitediff']):
    """Run comparison of multiple methods"""
    
    results = {}
    
    method_classes = {
        'savgol': SavitzkyGolayApproximator,
        'fourier': FourierApproximator,
        'chebyshev': ChebyshevApproximator,
        'finitediff': FiniteDifferenceApproximator
    }
    
    for method_name in methods:
        if method_name in method_classes:
            print(f"Running {method_name}...")
            
            try:
                approximator = method_classes[method_name](t, y)
                result = approximator.evaluate(t_eval)
                results[method_name] = result
                print(f"  ✓ {method_name} completed")
            except Exception as e:
                print(f"  ✗ {method_name} failed: {e}")
                results[method_name] = None
    
    return results

def main():
    """Command line interface for testing"""
    parser = argparse.ArgumentParser(description='Test Python derivative approximation methods')
    parser.add_argument('--input', required=True, help='Input CSV file with t,y columns')
    parser.add_argument('--output', required=True, help='Output JSON file for results')
    parser.add_argument('--methods', nargs='+', default=['savgol', 'fourier', 'chebyshev'], 
                       help='Methods to test')
    
    args = parser.parse_args()
    
    # Load data
    data = pd.read_csv(args.input)
    t = data['t'].values
    y = data['y'].values
    
    # Use same points for evaluation
    t_eval = t
    
    # Run comparison
    results = run_method_comparison(t, y, t_eval, args.methods)
    
    # Save results
    with open(args.output, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for method, result in results.items():
            if result is not None:
                json_results[method] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                      for k, v in result.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()