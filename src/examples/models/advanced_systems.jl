# advanced_systems.jl
# Advanced ODE systems for derivative approximation benchmarking

using ODEParameterEstimation

"""
    van_der_pol()

Create a Van der Pol oscillator for benchmarking.
Uses the pre-defined vanderpol function from ODEParameterEstimation.
"""
function van_der_pol()
    return vanderpol()
end

"""
    brusselator()

Create a Brusselator reaction-diffusion system for benchmarking.
Uses the pre-defined brusselator function from ODEParameterEstimation.
"""
function brusselator()
    return ODEParameterEstimation.brusselator()
end