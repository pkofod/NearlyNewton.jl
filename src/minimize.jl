include("linesearch/ls_optimize.jl")


function minimize(f∇f::T1, x0, scheme, B0, options::OptOptions=OptOptions()) where {T1}
# function minimize(f::F1, ∇f::F2, x0, scheme, approx, B0) where {F1, F2}

    linesearch = BackTracking()

    cache = preallocate_qn_caches(x0)

    g_tol, max_iter, show_trace = options.g_tol, options.max_iter, options.show_trace

    # Maintain current state in x_curr
    cache.x_curr = x0

    # Update current gradient
    f_curr, cache.∇f_curr = f∇f(true, cache.x_curr)

    f_next, B = iterate(cache, scheme, linesearch, f∇f, f_curr, B0, true; g_tol=options.g_tol)

    iter = 0
    while iter <= max_iter
        iter += 1

        f_curr = f_next
        f_next, B = iterate(cache, scheme, linesearch, f∇f, f_curr, B; g_tol=options.g_tol)

        # Check for gradient convergence
        if norm(cache.∇f_next) < g_tol || any(isnan.(cache.x_next))
            return cache.x_next, cache.∇f_next, iter
        end
    end

    return cache.x_next, cache.∇f_next, iter
end

function iterate(cache, scheme, linesearch::LineSearch, f∇f::T, f_curr, B, first=false; g_tol=1e-8) where T
    # This just moves all "next"s into "curr"s.
    shift!(cache)

    # Update current gradient and calculate the search direction
    cache.d = find_direction(B, cache.∇f_curr, scheme) # solve Bd = -∇f

    # # Perform line search along d
    α, f_α, ls_success = linesearch(f∇f, cache.x_curr, cache.d, f_curr, cache.∇f_curr, 1.0)
    # return α, f_α, ls_success

    # # Calculate final step vector and update the state
    cache.s = @. α * cache.d
    cache.x_next = @. cache.x_curr + cache.s

    # # Update gradient
    f_next, cache.∇f_next = f∇f(cache.∇f_next, cache.x_next)

    cache.y = cache.∇f_next - cache.∇f_curr

    B = update_qn(cache, B, scheme, first)

    return f_next, B
end

function trace_show(show_trace, f_curr, f_next, x_next, x_curr, α)
    if show_trace
        println("Objective value (curr): ", f_next)
        println("Objective value (prev): ", f_curr)
        println("Step size: ", α)
    end
end
