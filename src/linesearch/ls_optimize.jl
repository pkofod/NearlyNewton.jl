abstract type LineSearch end
include("linesearches/backtracking.jl")

function minimize!(f∇f!, x0, scheme::QuasiNewton, B0=I, options=OptOptions(), cache=preallocate_qn_caches_inplace(x0))
    minimize!(f∇f!, x0, (scheme, BackTracking()), B0, options, cache )
end
function minimize!(objective, x0, approach::Tuple{<:Any, <:LineSearch}, B0=I,
                   options::OptOptions=OptOptions(), # options
                   cache = preallocate_qn_caches_inplace(x0), # preallocate arrays for QN
                   )

    # first evaluation
    f, ∇f = objective(cache.∇fz, x0)

    # first iteration
    x, f, ∇f, B, is_converged = iterate!(cache, f, B0, approach, objective, options)

    iter = 0
    while iter <= options.max_iter && !is_converged
        iter += 1

        # take a step and update approximation
        x, f, ∇f, B, is_converged = iterate!(cache, f, B, approach, objective, options, false)
    end
    return x, f, ∇f, iter
end

function iterate!(cache, fx, B, approach, f∇f!, options, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach

    ∇fx, ∇fz, y, x, z, d, s = cache.∇fx, cache.∇fz, cache.y, cache.x, cache.z, cache.d, cache.s

    # This just moves all "next"s into "curr"s.
    copyto!(∇fx, ∇fz)
    copyto!(x, z)

    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, ∇fx, scheme) # solve Bd = -∇f

    # Perform line search along d
    α, f_α, ls_success = linesearch(f∇f!, d, x, fx, ∇fx, 1.0)

    # Calculate final step vector and update the state
    @. s = α * d
    @. z = x + s

    # Update gradient
    fz, ∇fz = f∇f!(∇fz, z)

    B = update_qn!(cache, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(cache.z, cache.∇fz, options.g_tol)

    return z, fz, ∇fz, B, is_converged
end


function minimize(f∇f!, x0, scheme::QuasiNewton, B0=I, options=OptOptions())
    minimize(f∇f!, x0, (scheme, BackTracking()), B0, options)
end
function minimize(objective::T1, x0, approach::Tuple{<:Any, <:LineSearch}, B0=I,
                  options::OptOptions=OptOptions(),
                  linesearch::T2 = BackTracking()
                  ) where {T1, T2}
    # first evaluation
    fx, ∇fx = objective(true, x0)

    # first iteration
    z, fz, ∇fz, B, is_converged = iterate(x0, fx, ∇fx, B0, approach, objective, options)

    iter = 0
    while iter <= options.max_iter && !is_converged
        iter += 1

        # take a step and update approximation
        z, fz, ∇fz, B, is_converged = iterate(z, fz, ∇fz, B, approach, objective, options, false)
    end

    return z, fz, ∇fz, iter
end

function iterate(x, fx, ∇fx, B, approach, f∇f, options, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach

    # Update current gradient and calculate the search direction
    d = find_direction(B, ∇fx, scheme) # solve Bd = -∇f

    # # Perform line search along d
    α, f_α, ls_success = linesearch(f∇f, d, x, fx, ∇fx, 1.0)

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s

    # # Update gradient
    fz, ∇fz = f∇f(∇fx, z)

    B = update_qn(d, s, ∇fx, ∇fz, B, scheme, is_first)

    # Check for gradient convergence
    is_converged = converged(z, ∇fz, options.g_tol)

    return z, fz, ∇fz, B, is_converged
end

function converged(z, ∇fz, g_tol)
    g_converged = norm(∇fz) < g_tol
    return g_converged || any(isnan.(z))
end
