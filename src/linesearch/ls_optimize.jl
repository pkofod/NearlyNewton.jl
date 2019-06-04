abstract type LineSearch end
include("linesearches/backtracking.jl")

function minimize!(f∇f!, x0, scheme::QuasiNewton, B0, options=OptOptions(), cache=preallocate_qn_caches_inplace(x0))
    minimize!(f∇f!, x0, (scheme, BackTracking()), B0, options, cache )
end

function minimize!(f∇f!, x0, approach::Tuple{<:Any, <:LineSearch}, B0,
                   options::OptOptions=OptOptions(), # options
                   cache = preallocate_qn_caches_inplace(x0), # preallocate arrays for QN
                   )
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach

    ∇f_curr, ∇f_next, y, x_curr, x_next, d, s = cache.∇f_curr, cache.∇f_next, cache.y, cache.x_curr, cache.x_next, cache.d, cache.s

    # first evaluation
    x_next .= x0
    f_curr = f∇f!(cache.∇f_next, cache.x_next)

    # first iteration
    f_next, B, is_converged = iterate!(cache, scheme, linesearch, f∇f!, f_curr, options, B0)

    iter = 0
    while iter <= options.max_iter && !is_converged
        iter += 1

        # save last objective value
        f_curr = f_next

        # take a step and update approximation
        f_next, B, is_converged = iterate!(cache, scheme, linesearch, f∇f!, f_curr, options, B, false)

    end
    return cache.x_next, cache.∇f_next, iter
end

function iterate!(cache, scheme, linesearch::LineSearch, f∇f!, f_curr, options, B, is_first=nothing)
    g_tol = options.g_tol
    ∇f_curr, ∇f_next, y, x_curr, x_next, d, s = cache.∇f_curr, cache.∇f_next, cache.y, cache.x_curr, cache.x_next, cache.d, cache.s

    # This just moves all "next"s into "curr"s.
    shift!(cache)

    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, ∇f_curr, scheme) # solve Bd = -∇f

    # Perform line search along d
    α, f_α, ls_success = linesearch(f∇f!, x_curr, d, f_curr, ∇f_curr, 1.0)

    # Calculate final step vector and update the state
    @. s = α * d
    @. x_next = x_curr + s

    # Update gradient
    f_next = f∇f!(∇f_next, x_next)

    B = update_qn!(cache, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(cache.x_next, cache.∇f_next, options.g_tol)

    return f_next, B, is_converged
end

function converged(x_next, ∇f_next, g_tol)
    g_converged = norm(∇f_next) < g_tol
    return g_converged || any(isnan.(x_next))
end

function minimize(f∇f::T1, x0, scheme, B0=I, options::OptOptions=OptOptions(), linesearch::T2 = BackTracking()) where {T1, T2}

    # Maintain current state in x_curr
    x_next = copy(x0)

    # Update current gradient
    f_next, ∇f_next = f∇f(true, x_next)

    ∇f_curr = copy(∇f_next)
    x_curr = copy(x_next)
    f_curr = f_next
    x_next, f_next, ∇f_next, B, is_converged = iterate(∇f_curr, x_curr, scheme, linesearch, f∇f, f_next, B0, options)

    iter = 0
    while iter <= options.max_iter && !is_converged
        iter += 1

        f_curr = f_next
        ∇f_curr = copy(∇f_next)
        x_curr = copy(x_next)
        x_next, f_next, ∇f_next, B, is_converged = iterate(∇f_curr, x_curr, scheme, linesearch, f∇f, f_curr, B, options, false)

    end

    return x_next, ∇f_next, iter
end

function iterate(∇f_curr, x_curr, scheme, linesearch::LineSearch, f∇f, f_curr, B, options, is_first=nothing)
    # Update current gradient and calculate the search direction
    d = find_direction(B, ∇f_curr, scheme) # solve Bd = -∇f

    # # Perform line search along d
    α, f_α, ls_success = linesearch(f∇f, x_curr, d, f_curr, ∇f_curr, 1.0)
    # return α, f_α, ls_success

    # # Calculate final step vector and update the state
    s = @. α * d
    x_next = @. x_curr + s

    # # Update gradient
    f_next, ∇f_next = f∇f(∇f_curr, x_next)

    B = update_qn(d, s, ∇f_curr, ∇f_next, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(x_next, ∇f_next, options.g_tol)
    
    return x_next, f_next, ∇f_next, B, is_converged
end
