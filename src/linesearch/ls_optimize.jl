function minimize(objective, x0, scheme::QuasiNewton, B0=nothing, options=OptOptions())
    minimize(objective, x0, (scheme, BackTracking()), B0, options)
end
function minimize(objective::T1, x0, approach::Tuple{<:Any, <:LineSearch}, B0=nothing,
                  options::OptOptions=OptOptions(),
                  linesearch::T2 = BackTracking()
                  ) where {T1, T2}
    T = eltype(x0)
    scheme, linesearch = approach

    x, fx, ∇fx, z, fz, ∇fz, B = prepare_variables(objective, approach, x0, copy(x0), B0)

    # first iteration
    z, fz, ∇fz, B, is_converged = iterate(z, fx, ∇fx, B, approach, objective, options)

    iter = 0
    while iter <= options.max_iter && !is_converged
        iter += 1

        # take a step and update approximation
        z, fz, ∇fz, B, is_converged = iterate(z, fz, ∇fz, B, approach, objective, options, false)
    end

    return z, fz, ∇fz, iter
end

function iterate(x, fx, ∇fx, B, approach, obj, options, is_first=nothing)
    # split up the approach into the hessian approximation scheme and line search
    scheme, linesearch = approach

    # Update current gradient and calculate the search direction
    d = find_direction(B, ∇fx, scheme) # solve Bd = -∇f

    # # Perform line search along d
    α, f_α, ls_success = linesearch(obj, d, x, fx, ∇fx, 1.0)

    # # Calculate final step vector and update the state
    s = @. α * d
    z = @. x + s

    # # Update gradient
    fz, ∇fz = obj(∇fx, z)

    if isa(scheme, Newton)
        fz, ∇fz, B = objective(B, ∇fz, z)
    else
        fz, ∇fz, B = update_qn(obj, d, s, ∇fx, z, ∇fz, B, scheme, is_first)
    end
    # Check for gradient convergence
    is_converged = converged(z, ∇fz, options.g_tol)

    return z, fz, ∇fz, B, is_converged
end
