# builtin_examples.jl

"""
    load_builtin_example(name::String)

Load a built-in ODE example for benchmarking.
Available examples:
- "lv_periodic": Lotka-Volterra with periodic dynamics
- "sir": SIR epidemic model
- "biomd6": BioModels database example 6
- "simple_oscillator": Harmonic oscillator
"""
function load_builtin_example(name::String)
    # For now, load from ODEParameterEstimation's examples
    # In a standalone version, these would be defined here
    
    examples_file = joinpath(dirname(dirname(@__DIR__)), "src", "examples", "load_examples.jl")
    include(examples_file)
    
    if name == "lv_periodic"
        return lv_periodic()
    elseif name == "sir"
        return SIR()
    elseif name == "biomd6"
        return BIOMD6()
    elseif name == "simple_oscillator"
        return simple_oscillator()
    else
        error("Unknown example: $name. Available: lv_periodic, sir, biomd6, simple_oscillator")
    end
end

# Define minimal examples directly for standalone use
# (These would be expanded in a production version)

"""
    create_lotka_volterra_example()

Create a Lotka-Volterra predator-prey model for benchmarking.
"""
function create_lotka_volterra_example()
    @parameters α=1.5 β=1.0 γ=3.0 δ=1.0
    @variables t x(t) y(t)
    D = Differential(t)
    
    eqs = [
        D(x) ~ α * x - β * x * y
        D(y) ~ -γ * y + δ * x * y
    ]
    
    @named sys = ODESystem(eqs)
    
    measured_quantities = [
        x ~ x,
        y ~ y
    ]
    
    ic = OrderedDict(x => 1.0, y => 1.0)
    p_true = OrderedDict(α => 1.5, β => 1.0, γ => 3.0, δ => 1.0)
    p_init = OrderedDict(α => 1.2, β => 0.8, γ => 2.5, δ => 0.8)
    
    return ParameterEstimationProblem(
        sys,
        measured_quantities,
        [0.0, 10.0],
        p_true,
        p_init,
        ic
    )
end

"""
    create_harmonic_oscillator_example()

Create a simple harmonic oscillator for benchmarking.
"""
function create_harmonic_oscillator_example()
    @parameters k=1.0 m=1.0 c=0.1
    @variables t x(t) v(t)
    D = Differential(t)
    
    eqs = [
        D(x) ~ v
        D(v) ~ -(k/m) * x - (c/m) * v
    ]
    
    @named sys = ODESystem(eqs)
    
    measured_quantities = [
        x ~ x,
        v ~ v
    ]
    
    ic = OrderedDict(x => 1.0, v => 0.0)
    p_true = OrderedDict(k => 1.0, m => 1.0, c => 0.1)
    p_init = OrderedDict(k => 0.8, m => 1.2, c => 0.15)
    
    return ParameterEstimationProblem(
        sys,
        measured_quantities,
        [0.0, 20.0],
        p_true,
        p_init,
        ic
    )
end