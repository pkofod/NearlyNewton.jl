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
    f_next, B = iterate!(cache, scheme, linesearch, f∇f!, f_curr, options, B0)

    # Check for gradient convergence
    if converged(cache, options.g_tol)
        return cache.x_next, cache.∇f_next, iter
    end

    iter = 0
    while iter <= options.max_iter
        iter += 1

        # save last objective value
        f_curr = f_next

        # take a step and update approximation
        f_next, B = iterate!(cache, scheme, linesearch, f∇f!, f_curr, options, B, false)

        # Check for gradient convergence
        if converged(cache, options.g_tol)
            return cache.x_next, cache.∇f_next, iter
        end
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

    return f_next, B
end

function converged(cache, g_tol)
    g_converged = norm(cache.∇f_next) < g_tol
    return g_converged || any(isnan.(cache.x_next))
end
