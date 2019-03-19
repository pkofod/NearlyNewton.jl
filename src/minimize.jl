function preallocate_minimize_caches(x0, inverse, scheme)
    n = length(x0)

    # Maintain current state in x_curr
    x_curr = copy(x0)

    B = initial_approximation(inverse, scheme, n)

    # Maintain next and current gradients in ∇f_next and ∇f_curr
    ∇f_next, ∇f_curr = similar(x0), similar(x0)

    # Maintain the search direction in d
    d = similar(x0)

    Δx = similar(x0)
    x_next = similar(x0)
    y = similar(∇f_next)

    return x_curr, ∇f_next, ∇f_curr, B, d, Δx, x_next, y
end
function minimize!(f, ∇f!, scheme, x0; inverse=true, c=0.001, g_tol=1e-8, max_iter=10^6, ls_max_iter=50, show_trace=false,  show_ls_trace=false)

    x_curr, ∇f_next, ∇f_curr, B, d, Δx, x_next, y = preallocate_minimize_caches(x0, inverse, scheme)

    # Update gradient
    ∇f!(∇f_next, x_next)
    x_next, ∇f_next = iterate!(scheme, f, ∇f!, ∇f_curr, ∇f_next, x_curr, x_next, d, B, Δx, y; c=c, show_ls_trace=show_ls_trace, ls_max_iter=ls_max_iter, g_tol=g_tol)

    # Check for gradient convergence
    converged = norm(∇f_next) < g_tol

    if converged
        return x_next, ∇f_next, 0
    end

    iter = 0
    while iter <= max_iter
        iter += 1

        x_next, ∇f_next = iterate!(scheme, f, ∇f!, ∇f_curr, ∇f_next, x_curr, x_next, d, B, Δx, y; c=c, show_ls_trace=show_ls_trace, ls_max_iter=ls_max_iter, g_tol=g_tol)

        # Check for gradient convergence
        converged = norm(∇f_next) < g_tol

        if converged
            return x_next, ∇f_next, iter
        end
    end
    return x_next, ∇f_next, iter
end

function iterate!(scheme, f, ∇f!, ∇f_curr, ∇f_next, x_curr, x_next, d, B, Δx, y; c=0.001, show_ls_trace=false, ls_max_iter=50, g_tol=1e-8)

    copyto!(x_curr, x_next)
    copyto!(∇f_curr, ∇f_next)
    
    # Update current gradient and calculate the search direction
    d = find_direction!(d, B, ∇f_curr) # solve Bd = -∇f

    # Perform line search along d
    α, f_α, ls_success = backtracking(f, x_curr, d, ∇f_curr; α_0=1.0, c=c, verbose=show_ls_trace, ls_max_iter=ls_max_iter)

    # Calculate final step vector and update the state
    @. Δx = α * d
    @. x_next = x_curr + Δx

    # Update gradient
    ∇f!(∇f_next, x_next)

    # Update y
    @. y = ∇f_next - ∇f_curr

    # Quasi-Newton update
    update!(scheme, B, Δx, y)

    return x_next, ∇f_next
end

function minimize(f, ∇f, scheme, x0; inverse=true, c=0.001, g_tol=1e-8, max_iter=10^6, ls_max_iter=50, show_trace=false, show_ls_trace=false)
    n = length(x0)

    # Maintain current state in x_curr
    x_curr = copy(x0)

    # Update current gradient
    ∇f_curr = ∇f(x_curr)

    B = initial_approximation(inverse, scheme, n)

    d = find_direction(B, ∇f_curr)

    # Perform a backtracking step
    α, f_α, ls_success = backtracking(f, x_curr, d, ∇f_curr; α_0=1.0, c=c, verbose=show_ls_trace, ls_max_iter=ls_max_iter)

    # Maintain the change in x in Δx
    Δx = α * d

    # Maintain the new x in x_next
    x_next = x_curr + Δx

    # Update the gradient at the new point
    ∇f_next = ∇f(x_next)

    # Update y
    y = ∇f_next - ∇f_curr

    # Update the approximation
    update!(scheme, B, Δx, y)

    iter = 0
    while iter <= max_iter
        iter += 1
        x_curr = copy(x_next)
        ∇f_curr = ∇f(x_curr)

        # Solve the system Bd = -∇f to find the search direction d
        d = find_direction(B, ∇f_curr)

        # Perform line search along d
        α, f_α, ls_success = backtracking(f, x_curr, d, ∇f_curr; α_0=1.0, c=c, verbose=show_ls_trace, ls_max_iter=ls_max_iter)

        # Update change in x according to α from above
        Δx = α .* d

        # Take step
        x_next = x_curr + Δx

        # Update gradient
        ∇f_next = ∇f(x_next)

        # Update y
        y = ∇f_next - ∇f_curr

        # Check for gradient convergence
        if norm(∇f_next) < g_tol
            return x_next, ∇f_next, iter
        end

        # Check for gradient convergence
        if show_trace
            println("Objective value (curr): ", f(x_next))
            println("Objective value (prev): ", f(x_curr))
            println("Step size: ", α)
        end

        # Quasi-Newton update
        update!(scheme, B, Δx, y)
    end
    return x_next, ∇f_next, iter
end
