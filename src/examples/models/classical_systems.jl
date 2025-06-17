# classical_systems.jl
# Classical ODE systems for derivative approximation benchmarking

using ODEParameterEstimation

"""
    lv_periodic()

Create a Lotka-Volterra predator-prey model with periodic behavior for benchmarking.
Uses the pre-defined lotka_volterra function from ODEParameterEstimation.
"""
function lv_periodic()
    return lotka_volterra()
end

"""
    harmonic_oscillator()

Create a harmonic oscillator system for benchmarking.
Uses the pre-defined harmonic function from ODEParameterEstimation.
"""
function harmonic_oscillator()
    return harmonic()
end