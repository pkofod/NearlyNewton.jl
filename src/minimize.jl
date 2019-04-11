struct OptOptions{T1, T2}
    c::T1
    g_tol::T1
    max_iter::T2
    show_trace::Bool
end

OptOptions(; c=1e-4, g_tol=1e-8, max_iter=10^4, show_trace=false) =
    OptOptions(c, g_tol, max_iter, show_trace)

function preallocate_minimize_caches(x0)
    n = length(x0)

    # Maintain current state in x_curr
    x_curr = copy(x0)

    # Maintain next and current gradients in ∇f_next and ∇f_curr
    ∇f_next, ∇f_curr = similar(x0), similar(x0)

    # Maintain the search direction in d
    d = similar(x0)

    Δx = similar(x0)
    x_next = similar(x0)
    y = similar(∇f_next)

    return x_curr, ∇f_next, ∇f_curr, d, Δx, x_next, y
end
function minimize!(f∇f!, x0, scheme, approx, B0, options::OptOptions)
    c, g_tol, max_iter, show_trace = options.c, options.g_tol, options.max_iter, options.show_trace

    x_curr, ∇f_next, ∇f_curr, d, Δx, x_next, y = preallocate_minimize_caches(x0)

    # Update gradient
    x_next .= x0
    f_curr, ∇f_next = f∇f!(∇f_next, x_next)
    x_next, f_next, ∇f_next, B = iterate!(scheme, approx, f∇f!, f_curr, ∇f_curr, ∇f_next, x_curr, x_next, d, B0, Δx, y; first=true, c=c, g_tol=g_tol)

    # Check for gradient convergence
    converged = norm(∇f_next) < g_tol

    if converged
        return x_next, ∇f_next, 0
    end

    iter = 0
    while iter <= max_iter
        iter += 1
        f_curr = f_next
        x_next, f_next, ∇f_next, B = iterate!(scheme, approx, f∇f!, f_curr, ∇f_curr, ∇f_next, x_curr, x_next, d, B, Δx, y; c=c, g_tol=g_tol)
        # Check for gradient convergence
        converged = norm(∇f_next) < g_tol

        if converged
            return x_next, ∇f_next, iter
        end
    end
    return x_next, ∇f_next, iter
end

function iterate!(scheme, approx, f∇f!, f_curr, ∇f_curr, ∇f_next, x_curr, x_next, d, B, Δx, y; first=false, c=0.001, g_tol=1e-8)
    copyto!(x_curr, x_next)
    copyto!(∇f_curr, ∇f_next)

    # Update current gradient and calculate the search direction
    d = find_direction!(d, scheme, approx, B, ∇f_curr) # solve Bd = -∇f
    lsoptions = LSOptions(0.5, 1e-4, 100, true)
    # Perform line search along d
    α, f_α, ls_success = backtracking(f∇f!, x_curr, d, f_curr, ∇f_curr, 1.0, lsoptions)
    # Calculate final step vector and update the state
    @. Δx = α * d
    @. x_next = x_curr + Δx
    # Update gradient
    f_next, ∇f_next = f∇f!(∇f_next, x_next)
    # Update y
    @. y = ∇f_next - ∇f_curr
    if first
        Badj = dot(y, d)/dot(y, y)*I
    else
        Badj = B
    end
    # Quasi-Newton update
    B = update!(scheme, approx, Badj, Δx, y)

    return x_next, f_next, ∇f_next, B
end

function minimize(f∇f::T1, x0, scheme, approx, B0, options::OptOptions) where {T1}
# function minimize(f::F1, ∇f::F2, x0, scheme, approx, B0) where {F1, F2}
    c, g_tol, max_iter, show_trace = options.c, options.g_tol, options.max_iter, options.show_trace
    lsoptions = LSOptions(0.5, 1e-4, 100, true)

    # Maintain current state in x_curr
    x_curr = x0
    # Update current gradient
    f_curr, ∇f_curr = f∇f(true, x_curr)
    # Find direction using Hessian approximation
    d = find_direction(scheme, approx, B0, ∇f_curr)
    # Perform a backtracking step
    α, f_α, ls_success = backtracking(f∇f, x_curr, d, f_curr, ∇f_curr, 1.0, lsoptions)
    # Maintain the change in x in Δx
    Δx = α * d
    # Maintain the new x in x_next
    x_next = x_curr + Δx
    # Update the gradient at the new point
    f_next, ∇f_next = f∇f(true, x_next)
    # Update y
    y = ∇f_next - ∇f_curr
    # Badj = α*I
    Badj = dot(y, d)/dot(y, y)*I

    # Update the approximation
    B = update(scheme, approx, Badj, Δx, y)
    iter = 0
    while iter <= max_iter
        iter += 1
        x_curr = copy(x_next)
        f_curr = f_next
        ∇f_curr = ∇f_next

        # Solve the system Bd = -∇f to find the search direction d
        d = find_direction(scheme, approx, B, ∇f_curr)
        # Perform line search along d
        α, f_α, ls_success = backtracking(f∇f, x_curr, d, f_curr, ∇f_curr, 1.0, lsoptions)
        # Update change in x according to α from above
        Δx = α * d

        # Take step
        x_next = x_curr + Δx
        # Update gradient
        f_next, ∇f_next = f∇f(true, x_next)
        # # Update y
        y = ∇f_next - ∇f_curr

        trace_show(show_trace, f_curr, f_next, x_next, x_curr, α)

        # Check for gradient convergence
        if norm(∇f_next) < g_tol
            return x_next, ∇f_next, iter
        end

        # Quasi-Newton update
        B = update(scheme, approx, B, Δx, y)

    end
    return x_next, ∇f_next, iter
end

function trace_show(show_trace, f_curr, f_next, x_next, x_curr, α)
    if show_trace
        println("Objective value (curr): ", f_next)
        println("Objective value (prev): ", f_curr)
        println("Step size: ", α)
    end
end
