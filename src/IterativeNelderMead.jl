module IterativeNelderMead

# Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)

using Statistics, LinearAlgebra

export optimize

struct NMOptions
    max_fcalls::Int
    no_improve_break::Int
    n_iterations::Int
    ftol_rel::Float64
    ftol_abs::Float64
    xtol_rel::Float64
    xtol_abs::Float64
end

struct Subspace
    index::Union{Int, Nothing}
    indices::Vector{Int}
    indicesv::Vector{Int}
end

function NMOptions(n_vary::Int, ;max_fcalls::Int=50_000 * n_vary, no_improve_break::Int=3, n_iterations::Int=n_vary, ftol_rel::Real=1E-8, ftol_abs::Real=1E-12, xtol_rel::Real=1E-8, xtol_abs::Real=1E-12)
    @assert n_vary > 0
    return NMOptions(max_fcalls, no_improve_break, n_iterations, ftol_rel, ftol_abs, xtol_rel, xtol_abs)
end

mutable struct NMState
    const p0::Vector{Float64}
    const lb::Vector{Float64}
    const ub::Vector{Float64}
    const vary::Vector{Bool}
    const pbest::Vector{Float64}
    const ptest::Vector{Float64}
    const simplex′::Matrix{Float64}
    fprev::Float64
    fbest::Float64
    iteration::Int
    fcalls::Int
    subspace::Subspace
end


function initial_state(obj, p0, lb, ub, vary)
    simplex′ = initial_simplex(p0, lb, ub, vary)
    fbest = obj(p0)
    @assert isfinite(fbest) "Objective $(obj) is not finite at initial value $(p0)"
    inds = findall(vary)
    indsv = collect(1:length(p0))
    return NMState(p0, lb, ub, vary, copy(p0), copy(p0), simplex′, fbest, fbest, 1, 0, Subspace(nothing, inds, indsv))
end


function initial_simplex(p0, lb, ub, vary)
    scale_factors = get_scale_factors(p0, lb, ub, vary, factor=0.15)
    indsv = findall(vary)
    nv = length(indsv)
    p0v = @view p0[indsv]
    lbv = @view lb[indsv]
    ubv = @view ub[indsv]
    scale_factorsv = @view scale_factors[indsv]
    simplex = repeat(p0v, 1, nv+1)
    simplex[:, 1:end-1] .+= diagm(scale_factorsv)
    for i=1:nv
        clamp!(view(simplex, i, :), lbv[i], ubv[i])
    end
    simplex′ = copy(simplex)
    for i=1:nv+1
        simplex′[:, i] .= @views param2unbounded.(simplex[:, i], lbv, ubv)
    end
    return simplex′
end


function get_subspaces(vary)
    subspaces = Subspace[]
    vi = findall(vary)
    nv = length(vi)
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
    return subspaces
end


function optimize(obj, p0::Vector{<:Real};
        lower_bounds::Union{Vector{<:Real}, Nothing}=nothing,
        upper_bounds::Union{Vector{<:Real}, Nothing}=nothing,
        vary::Union{Vector{Bool}, Nothing}=nothing,
        options::Union{NamedTuple, NMOptions, Nothing}=nothing
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
    else
        vary = copy(vary)
    end

    # Sanity check bounds and vary
    for i in eachindex(vary)
        if lower_bounds[i] == upper_bounds[i]
            vary[i] = false
        end
    end

    # How many parameters to optimize
    indsv = findall(vary)
    nv = length(indsv)
    @assert nv > 0 "No parameters found to optimize."

    # Options
    if isnothing(options)
        options = NMOptions(sum(vary))
    elseif options isa NamedTuple
        options = NMOptions(sum(vary); options...)
    else
        @assert options isa NMOptions
    end

    # Initial state
    state = initial_state(obj, p0, lower_bounds, upper_bounds, vary)
    full_space = state.subspace

    # Get remainins subspaces
    subspaces = get_subspaces(vary)

    #Main.infiltrate(@__MODULE__, Base.@locals, @__FILE__, @__LINE__)

    # Loop over iterations
    for i=1:options.n_iterations

        # Perform Ameoba call for all parameters
        optimize!(obj, state, options)
        
        # If there's <= 2 params, a three-simplex is the smallest simplex used and only used once.
        if nv <= 2
            break
        end

        # Check fcalls
        if state.fcalls >= options.max_fcalls
            break
        end
        
        # Perform Ameoba call for subspaces
        for subspace ∈ subspaces
            state.subspace = subspace
            optimize!(obj, state, options)
        end

        # Check x
        dx_abs_converged = compute_dx_abs(state) < options.xtol_abs
        dx_rel_converged = compute_dx_rel(state) < options.xtol_rel

        # Check f
        df_abs_converged = compute_df_abs(state) < options.ftol_abs
        df_rel_converged = compute_df_rel(state) < options.ftol_rel

        # Check f calls
        fcalls_converged = state.fcalls >= options.max_fcalls

        # Converged?
        if dx_abs_converged || dx_rel_converged || df_abs_converged || df_rel_converged || fcalls_converged
            # @show compute_df_abs(state)
            # @show compute_df_rel(state)
            # @show compute_dx_rel(state)
            # @show compute_dx_abs(state)
            # @show state.fcalls
            # @show dx_abs_converged dx_rel_converged df_abs_converged df_rel_converged fcalls_converged
            break
        elseif i < options.n_iterations
            state.subspace = full_space
            state.iteration = i + 1
        end

    end
    
    # Output
    result = get_result(state)

    # Return
    return result

end


function get_subspace_simplex(state::NMState)
    if !isnothing(state.subspace.index)
        inds = state.subspace.indices
        indsv = state.subspace.indicesv
        n = length(inds)
        S′ = zeros(n, n+1)
        S′[:, 1] .= @views param2unbounded.(state.p0[inds], state.lb[inds], state.ub[inds])
        v2 = param2unbounded.(state.pbest[inds], state.lb[inds], state.ub[inds])
        S′[:, 2] .= v2
        for i=3:n+1
            S′[:, i] .= v2
            j = i - 2
            S′[j, i] = param2unbounded(state.p0[inds[j]], state.lb[inds[j]], state.ub[inds[j]])
        end
    else
        S′ = copy(state.simplex′)
    end
    return S′
end


function optimize!(obj, state::NMState, options::NMOptions)

    # Simplex for this subspace
    S′ = get_subspace_simplex(state)
    nx, nxp1 = size(S′)

    # Alias options
    max_fcalls = options.max_fcalls
    ftol_rel = options.ftol_rel
    ftol_abs = options.ftol_abs
    no_improve_break = options.no_improve_break

    # Number of times converged in a row
    n_converged = 0

    # Initiate storage arrays
    fvals = zeros(nxp1)
    xr = zeros(nx)
    xbar = zeros(nx)
    xc = zeros(nx)
    xe = zeros(nx)
    xcc = zeros(nx)
    
    # Generate the fvals for the initial simplex
    penalty = 1000 * abs(state.fbest)
    for i=1:nxp1
        fvals[i] = @views compute_obj(obj, S′[:, i], state, penalty, options)
    end

    # Sort the fvals and then simplex
    inds = sortperm(fvals)
    S′ .= S′[:, inds]
    fvals .= fvals[inds]
    x1 = S′[:, 1]
    xn = S′[:, end-1]
    xnp1 = S′[:, end]
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
        #@show compute_df_abs(f1, fnp1)
        #@show compute_df_rel(f1, fnp1)
        if (compute_df_rel(f1, fnp1) < ftol_rel) || (compute_df_abs(f1, fnp1) < ftol_abs)
            n_converged += 1
        else
            n_converged = 0
        end
        if n_converged >= no_improve_break
            break
        end

        # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
        # We want to iteratively replace the worst vector with a better vector.
        
        # The "average" vector, ignoring the worst point
        # We first anchor points off this average Vector
        xbar .= @views vec(mean(S′[:, 1:end-1], dims=2))
        
        # The reflection point
        xr .= xbar .+ α .* (xbar .- xnp1)
        
        # Update the current testing parameter with xr
        fr = compute_obj(obj, xr, state, fmax, options)

        if fr < f1
            xe .= xbar .+ γ .* (xbar .- xnp1)
            fe = compute_obj(obj, xe, state, fmax, options)
            if fe < fr
                S′[:, end] .= xe
                fvals[end] = fe
            else
                S′[:, end] .= xr
                fvals[end] = fr
            end
        elseif fr < fn
            S′[:, end] .= xr
            fvals[end] = fr
        else
            if fr < fnp1
                xc .= xbar .+ σ .* (xbar .- xnp1)
                fc = compute_obj(obj, xc, state, fmax, options)
                if fc <= fr
                    S′[:, end] .= xc
                    fvals[end] = fc
                else
                    shrink = true
                end
            else
                xcc .= xbar .+ σ .* (xnp1 .- xbar)
                fcc = compute_obj(obj, xcc, state, fmax, options)
                if fcc < fvals[end]
                    S′[:, end] .= xcc
                    fvals[end] = fcc
                else
                    shrink = true
                end
            end
        end
        if shrink
            for j=2:nxp1
                S′[:, j] .= @views x1 .+ δ .* (S′[:, j] .- x1)
                fvals[j] = @views compute_obj(obj, S′[:, j], state, fmax, options)
            end
        end

        # Sort
        inds = sortperm(fvals)
        fvals .= fvals[inds]
        S′ .= S′[:, inds]
        x1 .= S′[:, 1]
        xn .= S′[:, end-1]
        xnp1 .= S′[:, end]
        f1 = fvals[1]
        fn = fvals[end-1]
        fnp1 = fvals[end]
    end


    # Sort
    inds = sortperm(fvals)
    fvals .= fvals[inds]
    S′ .= S′[:, inds]
    x1 .= S′[:, 1]
    xn .= S′[:, end-1]
    xnp1 .= S′[:, end]
    f1 = fvals[1]
    fn = fvals[end-1]
    fnp1 = fvals[end]
    
    # Update the full simplex and best fit parameters
    inds = state.subspace.indices
    indsv = state.subspace.indicesv
    state.pbest[inds] .= @views param2bounded.(x1, state.lb[inds], state.ub[inds])
    state.ptest .= copy(state.pbest)
    state.fprev = state.fbest
    state.fbest = f1
    if !isnothing(state.subspace.index)
        _inds = findall(state.vary)
        state.simplex′[:, state.subspace.index] .= @views param2unbounded.(state.pbest[_inds], state.lb[_inds], state.ub[_inds])
    else
        state.simplex′ .= copy(S′)
    end

end


###################
#### TOLERANCE ####
###################

compute_dx_rel(state::NMState) = compute_dx_rel(simplex2bounded(state))
function compute_dx_rel(simplex::Matrix{<:Real})
    a = minimum(simplex, dims=2)
    b = maximum(simplex, dims=2)
    c = (abs.(b) .+ abs.(a)) ./ 2
    clamp!(c, 0, Inf)
    r = abs.(b .- a) ./ c
    return maximum(r)
end

compute_dx_abs(state::NMState) = compute_dx_abs(simplex2bounded(state))
function compute_dx_abs(simplex::Matrix{<:Real})
    a = minimum(simplex, dims=2)
    b = maximum(simplex, dims=2)
    r = abs.(b .- a)
    return maximum(r)
end

function simplex2bounded(state::NMState)
    S = similar(state.simplex′)
    indsv = findall(state.vary)
    for i in axes(S, 2)
        S[:, i] .= @views param2bounded.(state.simplex′[:, i], state.lb[indsv], state.ub[indsv])
    end
    return S
end

compute_df_rel(state::NMState) = compute_df_rel(state.fbest, state.fprev)

function compute_df_rel(a::Real, b::Real)
    avg = (abs(a) + abs(b)) / 2
    return abs(a - b) / avg
end

compute_df_abs(state::NMState) = compute_df_abs(state.fbest, state.fprev)
compute_df_abs(a::Real, b::Real) = abs(a - b)


# Computing the objective
function compute_obj(obj, x′, state::NMState, fmax::Real, options::NMOptions; increase::Bool=true)
    if increase
        state.fcalls += 1
    end
    inds = state.subspace.indices
    state.ptest[inds] .= param2bounded.(x′, state.lb[inds], state.ub[inds])
    f = obj(state.ptest)
    if !isfinite(f)
        f = 1E6 * fmax
        if !isfinite(f)
            f = 1E6
        end
    end
    return f
end


# Scale factors for initial simplex
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


# Bounding parameters
function param2bounded(x, lo, hi, vary=true)
    if ~vary
        return x
    end
    has_lo = isfinite(lo)
    has_hi = isfinite(hi)
    if has_lo && has_hi
        return lo + (sin(x) + 1) * ((hi - lo) / 2)
    elseif has_lo
        return lo - 1 + sqrt(x^2 + 1)
    elseif has_hi
        return hi + 1 - sqrt(x^2 + 1)
    else
        return x
    end
end


function param2unbounded(x, lo, hi, vary=true)
    if ~vary
        return x
    end
    has_lo = isfinite(lo)
    has_hi = isfinite(hi)
    if has_lo && has_hi
        return asin(2 * (x - lo) / (hi - lo) - 1)
    elseif has_lo
        return sqrt((x - lo + 1)^2 - 1)
    elseif has_hi
        return sqrt((hi - x + 1)^2 - 1)
    else
        return x
    end
end

# Output
function get_result(state::NMState)
    simplex = simplex2bounded(state)
    return (;state.pbest, state.fbest, state.fcalls, simplex, iterations=state.iteration, p0=state.p0, state)
end


end
