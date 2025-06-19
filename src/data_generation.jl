# data_generation.jl

"""
    generate_datasets(pep::ParameterEstimationProblem, config::BenchmarkConfig)

Generate clean and noisy datasets for benchmarking, including true derivatives.
"""
function generate_datasets(pep::ParameterEstimationProblem, config::BenchmarkConfig)
    # Determine time interval
    time_interval = isnothing(config.time_interval) ? pep.recommended_time_interval : config.time_interval
    if isnothing(time_interval)
        time_interval = [0.0, 5.0]
    end
    
    # Calculate derivatives symbolically
    expanded_mq, obs_derivs = calculate_observable_derivatives(
        equations(pep.model.system),
        pep.measured_quantities,
        config.derivative_orders
    )
    
    # Create new ODESystem with derivative observables
    @named new_sys = ODESystem(
        equations(pep.model.system), 
        t; 
        observed = expanded_mq
    )
    
    # Create and solve ODE problem
    prob = ODEProblem(
        structural_simplify(new_sys), 
        pep.ic, 
        (time_interval[1], time_interval[2]), 
        pep.p_true
    )
    
    # Solve with high accuracy
    sol = solve(
        prob, 
        AutoVern9(Rodas4P()), 
        abstol = 1e-14, 
        reltol = 1e-14, 
        saveat = range(time_interval[1], time_interval[2], length = config.data_size)
    )
    
    # Extract clean data
    clean_data = OrderedDict{Any, Vector{Float64}}()
    clean_data["t"] = sol.t
    
    # Store observables
    obs_to_key = Dict()
    for mq in pep.measured_quantities
        key = Num(mq.rhs)
        clean_data[key] = sol[mq.lhs]
        obs_to_key[mq.lhs] = key
    end
    
    # Store derivatives
    derivatives = OrderedDict{Any, Vector{Float64}}()
    derivatives["t"] = sol.t
    
    for i in 1:length(pep.measured_quantities)
        obs_key = obs_to_key[pep.measured_quantities[i].lhs]
        for d in 1:config.derivative_orders
            derivatives["d$(d)_$obs_key"] = sol[obs_derivs[i, d]]
        end
    end
    
    # Generate noisy data
    noisy_data = generate_noisy_data(clean_data, config)
    
    return (
        clean = clean_data,
        noisy = noisy_data,
        derivatives = derivatives,
        measured_quantities = pep.measured_quantities,
        obs_derivs = obs_derivs,
        solution = sol
    )
end

"""
    generate_noisy_data(clean_data, config; rng=Random.GLOBAL_RNG)

Add noise to clean data according to configuration.
Uses the provided RNG for reproducible noise generation.
"""
function generate_noisy_data(clean_data::OrderedDict, config::BenchmarkConfig; rng=Random.GLOBAL_RNG)
    noisy_data = OrderedDict{Any, Vector{Float64}}()
    
    for (key, values) in clean_data
        if key == "t"
            noisy_data[key] = values
        else
            if config.noise_type == "additive"
                # Additive noise scaled by mean signal magnitude
                noise_scale = config.noise_level * mean(abs.(values))
                noise = noise_scale * randn(rng, length(values))
                noisy_data[key] = values + noise
            elseif config.noise_type == "multiplicative"
                # Multiplicative noise
                noise = 1 .+ config.noise_level * randn(rng, length(values))
                noisy_data[key] = values .* noise
            else
                error("Unknown noise type: $(config.noise_type)")
            end
        end
    end
    
    return noisy_data
end

"""
    calculate_observable_derivatives(equations, measured_quantities, nderivs)

Calculate symbolic derivatives of observables up to the specified order.
"""
function calculate_observable_derivatives(equations, measured_quantities, nderivs)
    # Create equation dictionary for substitution
    equation_dict = Dict(eq.lhs => eq.rhs for eq in equations)
    
    n_observables = length(measured_quantities)
    
    # Create symbolic variables for derivatives
    ObservableDerivatives = Symbolics.variables(:d_obs, 1:n_observables, 1:nderivs)
    
    # Initialize vector to store derivative equations
    SymbolicDerivs = Vector{Vector{Equation}}(undef, nderivs)
    
    # Calculate first derivatives
    SymbolicDerivs[1] = [
        ObservableDerivatives[i, 1] ~ substitute(
            expand_derivatives(D(measured_quantities[i].rhs)), 
            equation_dict
        ) 
        for i in 1:n_observables
    ]
    
    # Calculate higher order derivatives
    for j in 2:nderivs
        SymbolicDerivs[j] = [
            ObservableDerivatives[i, j] ~ substitute(
                expand_derivatives(D(SymbolicDerivs[j-1][i].rhs)), 
                equation_dict
            ) 
            for i in 1:n_observables
        ]
    end
    
    # Create new measured quantities with derivatives
    expanded_measured_quantities = copy(measured_quantities)
    append!(expanded_measured_quantities, vcat(SymbolicDerivs...))
    
    return expanded_measured_quantities, ObservableDerivatives
end