module IterativeNelderMead

using Statistics, LinearAlgebra

export NelderMeadOptions, optimize, initial_simplex, initial_state

struct NelderMeadOptions
    max_fcalls::Int
    no_improve_break::Int
    n_iterations::Int
    ftol_rel::Float64
    ftol_abs::Float64
    xtol_rel::Float64
    xtol_abs::Float64
end


function NelderMeadOptions(n_vary::Int, ;max_fcalls::Int=150000 * n_vary, no_improve_break::Int=3, n_iterations::Int=n_vary, ftol_rel::Real=1E-8, ftol_abs::Real=1E-12, xtol_rel::Real=1E-8, xtol_abs::Real=1E-12)
    @assert n_vary > 0
    return NelderMeadOptions(max_fcalls, no_improve_break, n_iterations, ftol_rel, ftol_abs, xtol_rel, xtol_abs)
end


struct Subspace
    index::Int
    indices::Vector{Int}
    indicesv::Vector{Int}
end


mutable struct NelderMeadState
    const full_space::Subspace
    subspaces::Vector{Subspace}
    subspace::Subspace
    const full_simplex::Matrix{Float64}
    const sub_simplex::Matrix{Float64}
    ptest::Vector{Float64}
    ftest::Float64
    const pbest::Vector{Float64}
    fprev::Float64
    fbest::Float64
    iteration::Int
    fcalls::Int
end

function initial_state(obj, p0, lower_bounds, upper_bounds, vary, scale_factors)
    full_space, subspaces = initialize_subspaces(vary)
    full_simplex = initial_simplex(p0, lower_bounds, upper_bounds, vary, scale_factors)
    sub_simplex = copy(full_simplex)
    ptest = copy(p0)
    ftest = obj(p0)
    pbest = copy(p0)
    fbest = ftest
    return NelderMeadState(full_space, subspaces, full_space, full_simplex, sub_simplex, ptest, ftest, pbest, fbest, fbest, 1, 0)
end

function initial_simplex(p0, lower_bounds, upper_bounds, vary, scale_factors)
    indsv = findall(vary)
    p0v = p0[indsv]
    lower_boundsv = lower_bounds[indsv]
    upper_boundsv = upper_bounds[indsv]
    scale_factorsv = scale_factors[indsv]
    nv = length(p0v)
    simplex = repeat(p0v, 1, nv+1)
    simplex[:, 1:end-1] .+= diagm(scale_factors[indsv])
    for i=1:nv
        clamp!(view(simplex, i, :), lower_boundsv[i], upper_boundsv[i])
    end
    return simplex
end


function initialize_subspaces(vary)
    subspaces = Subspace[]
    vi = findall(vary)
    nv = length(vi)
    full_subspace = Subspace(0, vi, [1:nv;])
    if nv > 2
        for i=1:nv-1
            k1 = vi[i]
            k2 = vi[i+1]
            push!(subspaces, Subspace(i, [k1, k2], [i, i+1]))
        end
        k1 = vi[1]
        k2 = vi[end]
        push!(subspaces, Subspace(nv, [k1, k2], [1, nv]))
        if nv > 3
            k1 = vi[2]
            k2 = vi[end-1]
            push!(subspaces, Subspace(nv+1, [k1, k2], [2, nv-1]))
        end
    end
    return full_subspace, subspaces
end


function optimize(obj, p0::Vector{<:Real};
        lower_bounds::Union{Vector{<:Real}, Nothing}=nothing,
        upper_bounds::Union{Vector{<:Real}, Nothing}=nothing,
        vary::Union{Vector{Bool}, Nothing}=nothing,
        scale_factors::Union{Vector{<:Real}, Nothing}=nothing,
        options::Union{NamedTuple, NelderMeadOptions, Nothing}=nothing
    )

    ########################
    #### Resolve params ####
    ########################

    # Number of total params
    n = length(p0)

    # Parameter bounds
    if isnothing(lower_bounds)
        lower_bounds = fill(-Inf, n)
    end
    if isnothing(upper_bounds)
        upper_bounds = fill(Inf, n)
    end

    # Varied params
    if isnothing(vary)
        vary = trues(n)
        bad = findall(lower_bounds .== upper_bounds)
        vary[bad] .= false
    end
    nv = sum(vary)
    @assert nv > 0 "No parameters found to optimize."

    # Sanity check bounds and vary
    for i in eachindex(vary)
        if lower_bounds[i] == upper_bounds[i]
            vary[i] = false
        end
    end

    # Scale factors
    if isnothing(scale_factors)
        scale_factors = get_scale_factors(p0, lower_bounds, upper_bounds, vary)
    end

    # Options
    if isnothing(options)
        options = NelderMeadOptions(sum(vary))
    elseif options isa NamedTuple
        options = NelderMeadOptions(sum(vary); options...)
    else
        @assert options isa NelderMeadOptions
    end

    # Initial state
    state = initial_state(obj, p0, lower_bounds, upper_bounds, vary, scale_factors)

    # Loop over iterations
    for i=1:options.n_iterations

        #Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)

        # Perform Ameoba call for all parameters
        state.subspace = state.full_space
        state.fprev = state.fbest
        optimize_space!(state, obj, p0, lower_bounds, upper_bounds, vary, options)
        
        # If there's <= 2 params, a three-simplex is the smallest simplex used and only used once.
        if nv <= 2
            break
        end

        # Converged?
        if state.fcalls >= options.max_fcalls
            break
        end
        
        # Perform Ameoba call for subspaces
        for subspace ∈ state.subspaces
            state.subspace = subspace
            optimize_space!(state, obj, p0, lower_bounds, upper_bounds, vary, options)
        end

        # Check x
        dx_abs_converged = compute_dx_abs(state.full_simplex) < options.xtol_abs
        dx_rel_converged = compute_dx_rel(state.full_simplex) < options.xtol_rel

        # Check f
        df_abs_converged = compute_df_abs(state.fbest, state.fprev) < options.ftol_abs
        df_rel_converged = compute_df_rel(state.fbest, state.fprev) < options.ftol_rel

        # Check f calls
        fcalls_converged = state.fcalls >= options.max_fcalls

        # Converged?
        if dx_abs_converged || dx_rel_converged || df_abs_converged || df_rel_converged || fcalls_converged
            # @show compute_df_abs(state.fbest, state.fprev)
            # @show compute_df_rel(state.fbest, state.fprev)
            # @show compute_dx_rel(state.full_simplex)
            # @show compute_dx_abs(state.full_simplex)
            # @show state.fcalls
            # @show dx_abs_converged dx_rel_converged df_abs_converged df_rel_converged fcalls_converged
            break
        else
            state.iteration = i
        end

    end
    
    # Output
    result = get_result(state)

    # Return
    return result

end


function get_subspace_simplex(subspace, p0, pbest)
    n = length(subspace.indices)
    simplex = zeros(n, n+1)
    simplex[:, 1] .= p0[subspace.indices]
    simplex[:, 2] .= pbest[subspace.indices]
    for i=3:n+1
        simplex[:, i] .= pbest[subspace.indices]
        j = i - 2
        simplex[j, i] = p0[j]
    end
    return simplex
end


function optimize_space!(state::NelderMeadState, obj, p0, lower_bounds, upper_bounds, vary, options::NelderMeadOptions)

    # Simplex for this subspace
    if state.subspace.index < 1
        simplex = copy(state.full_simplex)
    else
        simplex = get_subspace_simplex(state.subspace, p0, state.pbest)
    end
    nx, nxp1 = size(simplex)

    # Max f evals
    max_fcalls = options.max_fcalls
    ftol_rel = options.ftol_rel
    ftol_abs = options.ftol_abs

    # Keeps track of the number of times the solver thinks it has converged in a row.
    no_improve_break = options.no_improve_break
    n_converged = 0

    # Initiate storage arrays
    fvals = zeros(nxp1)
    xr = zeros(nx)
    xbar = zeros(nx)
    xc = zeros(nx)
    xe = zeros(nx)
    xcc = zeros(nx)
    
    # Generate the fvals for the initial simplex
    for i=1:nxp1
        fvals[i] = @views compute_obj(obj, simplex[:, i], 1000 * abs(state.fbest), state, lower_bounds, upper_bounds, vary, options)
    end

    # Sort the fvals and then simplex
    inds = sortperm(fvals)
    simplex .= simplex[:, inds]
    fvals .= fvals[inds]
    x1 = simplex[:, 1]
    xn = simplex[:, end-1]
    xnp1 = simplex[:, end]
    f1 = fvals[1]
    fn = fvals[end-1]
    fnp1 = fvals[end]

    # Hyper parameters
    α = 1.0
    γ = 2.0
    σ = 0.5
    δ = 0.5
    
    # Loop
    while true

        # fmax for penalizing objective when things go bad
        fmax = maximum(abs.(fvals))
            
        # Checks whether or not to shrink if all other checks "fail"
        shrink = false

        # break after max number function calls is reached.
        if state.fcalls >= max_fcalls
            break
        end

        # Break if f tolerance has been met no_improve_break times in a row
        #if (compute_df_rel(f1, fnp1) < ftol_rel) || (compute_df_abs(f1, fnp1) < ftol_abs)
        if compute_df_rel(f1, fnp1) > ftol_rel
            n_converged = 0
        else
            n_converged += 1
        end
        if n_converged >= no_improve_break
            break
        end

        # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
        # We want to iteratively replace the worst vector with a better vector.
        
        # The "average" vector, ignoring the worst point
        # We first anchor points off this average Vector
        xbar .= @views vec(mean(simplex[:, 1:end-1], dims=2))
        
        # The reflection point
        xr .= xbar .+ α .* (xbar .- xnp1)
        
        # Update the current testing parameter with xr
        fr = compute_obj(obj, xr, fmax, state, lower_bounds, upper_bounds, vary, options)

        if fr < f1
            xe .= xbar .+ γ .* (xbar .- xnp1)
            fe = compute_obj(obj, xe, fmax, state, lower_bounds, upper_bounds, vary, options)
            if fe < fr
                simplex[:, end] .= xe
                fvals[end] = fe
            else
                simplex[:, end] .= xr
                fvals[end] = fr
            end
        elseif fr < fn
            simplex[:, end] .= xr
            fvals[end] = fr
        else
            if fr < fnp1
                xc .= xbar .+ σ .* (xbar .- xnp1)
                fc = compute_obj(obj, xc, fmax, state, lower_bounds, upper_bounds, vary, options)
                if fc <= fr
                    simplex[:, end] .= xc
                    fvals[end] = fc
                else
                    shrink = true
                end
            else
                xcc .= xbar .+ σ .* (xnp1 .- xbar)
                fcc = compute_obj(obj, xcc, fmax, state, lower_bounds, upper_bounds, vary, options)
                if fcc < fvals[end]
                    simplex[:, end] .= xcc
                    fvals[end] = fcc
                else
                    shrink = true
                end
            end
        end
        if shrink
            for j=2:nxp1
                simplex[:, j] .= @views x1 .+ δ .* (simplex[:, j] .- x1)
                fvals[j] = @views compute_obj(obj, simplex[:, j], fmax, state, lower_bounds, upper_bounds, vary, options)
            end
        end

        # Sort
        inds = sortperm(fvals)
        fvals .= fvals[inds]
        simplex .= simplex[:, inds]
        x1 .= simplex[:, 1]
        xn .= simplex[:, end-1]
        xnp1 .= simplex[:, end]
        f1 = fvals[1]
        fn = fvals[end-1]
        fnp1 = fvals[end]
    end


    # Sort
    inds = sortperm(fvals)
    fvals .= fvals[inds]
    simplex .= simplex[:, inds]
    x1 .= simplex[:, 1]
    xn .= simplex[:, end-1]
    xnp1 .= simplex[:, end]
    f1 = fvals[1]
    fn = fvals[end-1]
    fnp1 = fvals[end]
    
    # Update the full simplex and best fit parameters
    state.pbest[state.subspace.indices] .= x1
    state.fbest = f1
    vi = findall(vary)
    if state.subspace.index < 1
        state.full_simplex .= copy(simplex)
    else
        state.full_simplex[:, state.subspace.index] .= state.pbest[vi]
    end

end


###################
#### TOLERANCE ####
###################
    
function compute_dx_rel(simplex::AbstractMatrix{Float64})
    a = minimum(simplex, dims=2)
    b = maximum(simplex, dims=2)
    c = (abs.(b) .+ abs.(a)) ./ 2
    bad = findall(c .< 0)
    c[bad] .= 1
    r = abs.(b .- a) ./ c
    return maximum(r)
end

function compute_dx_abs(simplex::AbstractMatrix{Float64})
    a = minimum(simplex, dims=2)
    b = maximum(simplex, dims=2)
    r = abs.(b .- a)
    return maximum(r)
end

function compute_df_rel(a, b)
    avg = (abs(a) + abs(b)) / 2
    return abs(a - b) / avg
end

function compute_df_abs(a, b)
    return abs(a - b)
end

###################
#### OBJECTIVE ####
###################

function compute_obj(obj, x, fmax, state::NelderMeadState, lower_bounds, upper_bounds, vary, options::NelderMeadOptions; increase::Bool=true)
    if increase
        state.fcalls += 1
    end
    state.ptest[state.subspace.indices] .= x
    f = obj(state.ptest)
    #f = penalize(f, fmax, state.ptest, state.subspace, lower_bounds, upper_bounds, options)
    f = penalize(f, fmax, state.ptest, state.subspace, lower_bounds, upper_bounds, options)
    if !isfinite(f)
        f = 1E6 * fmax
    end
    return f
end

function penalize(f, fmax, ptest, subspace, lower_bounds, upper_bounds, options)
    penalty_factor = 100 * fmax
    for i in eachindex(subspace.indices)
        j = subspace.indices[i]
        if ptest[j] < lower_bounds[j]
            f += penalty_factor
            f += penalty_factor * (lower_bounds[j] - ptest[j])
        end
        if ptest[j] > upper_bounds[j]
            f += penalty_factor
            f += penalty_factor * (ptest[j] - upper_bounds[j])
        end
    end
    return f
end


function get_scale_factors(p0, lower_bounds, upper_bounds, vary; factor::Real=0.15)
    scale_factors = fill(NaN, length(p0))
    for i in eachindex(p0)
        if vary[i]
            has_lower = isfinite(lower_bounds[i])
            has_upper = isfinite(upper_bounds[i])
            if has_lower && has_upper
                scale_factors[i] = (upper_bounds[i] - lower_bounds[i]) * factor
            elseif has_lower
                scale_factors[i] = (p0[i] - lower_bounds[i]) * factor
            elseif has_upper
                scale_factors[i] = (upper_bounds[i] - p0[i]) * factor
            else
                scale_factors[i] = 0.5 * abs(p0[i])
            end
        end
    end
    return scale_factors
end


################
#### OUTPUT ####
################


function get_result(state::NelderMeadState)
    return (;pbest=state.pbest, fbest=state.fbest, fcalls=state.fcalls, simplex=state.full_simplex, iterations=state.iteration)
end

#function param2bounded(x, lo, hi)

#end

# function param2unbounded(x, lo, hi)
#     has_low = isfinite(lo)
#     has_high = isfinite(hi)
#     if has_low && has_high
#         return 
#     end
# end


end
