# biological_systems.jl
# Biological ODE systems for derivative approximation benchmarking

using ODEParameterEstimation

"""
    sir()

Create a SEIR epidemic model for benchmarking.
Uses the pre-defined seir function from ODEParameterEstimation.
"""
function sir()
    return seir()
end

"""
    fitzhugh_nagumo()

Create a FitzHugh-Nagumo neuron model for benchmarking.
Uses the pre-defined fitzhugh_nagumo function from ODEParameterEstimation.
"""
function fitzhugh_nagumo()
    return ODEParameterEstimation.fitzhugh_nagumo()
end