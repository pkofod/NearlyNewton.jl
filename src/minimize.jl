include("linesearch/ls_optimize.jl")

using InteractiveUtils
function minimize(f∇f::T1, x0, scheme, B0=I, options::OptOptions=OptOptions(), linesearch::T2 = BackTracking()) where {T1, T2}
    @code_warntype preallocate_qn_caches(x0)
    cache = preallocate_qn_caches(x0)
    # Maintain current state in x_curr
    cache.x_next = x0

    # Update current gradient
    f_next, cache.∇f_next = f∇f(true, cache.x_next)

    f_next, B = iterate(cache, scheme, linesearch, f∇f, f_next, B0, options)

    iter = 0
    while iter <= options.max_iter
        iter += 1

        f_curr = f_next
        f_next, B = iterate(cache, scheme, linesearch, f∇f, f_curr, B, options, false)

        # Check for gradient convergence
        if norm(cache.∇f_next) < options.g_tol || any(isnan.(cache.x_next))
            return cache.x_next, cache.∇f_next, iter
        end
    end

    return cache.x_next, cache.∇f_next, iter
end

function iterate(cache, scheme, linesearch::LineSearch, f∇f, f_curr, B, options, is_first=nothing)
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

    B = update_qn(cache, B, scheme, is_first)

    return f_next, B
end

function trace_show(show_trace, f_curr, f_next, x_next, x_curr, α)
    if show_trace
        println("Objective value (curr): ", f_next)
        println("Objective value (prev): ", f_curr)
        println("Step size: ", α)
    end
end
