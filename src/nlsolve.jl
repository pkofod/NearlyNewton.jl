function nlsolve!(f, ∇f!, scheme, x0; inverse=true, c=0.001, g_tol=1e-8, maxiter=10^4, show_trace=false)
    n = length(x0)
    x_curr = copy(x0)
    ∇f_next, ∇f_curr = similar(x0), similar(x0)
    ∇f!(∇f_curr, x_curr)

    B = initial_approximation(inverse, scheme, n)

    d = .-solve(B, ∇f_curr)
    α, f_α, ls_success = backtracking(f, x_curr, d, ∇f_curr; α_0=1.0, c=c)
    Δx = α .* d
    x_next = x_curr .+ Δx

    ∇f!(∇f_next, x_next)
    y = ∇f_next .- ∇f_curr
    update!(scheme, B, Δx, y)

    for i = 1:maxiter
        copyto!(x_curr, x_next)
        ∇f!(∇f_curr, x_curr)

        d = solve!(d, B, ∇f_curr)

        α, f_α, ls_success = backtracking(f, x_curr, d, ∇f_curr; α_0=1.0, c=c, verbose=show_trace)
        Δx .= α .* d
        x_next .= x_curr .+ Δx
        ∇f!(∇f_next, x_next)
        y .= ∇f_next .- ∇f_curr

        if norm(∇f_next) < g_tol
            return x_next, ∇f_next
        end

        update!(scheme, B, Δx, y)
    end
    return x_next, ∇f_next
end


function nlsolve(f, ∇f, scheme, x0; inverse=true, c=0.001, g_tol=1e-8, maxiter=10^4, show_trace=false)
    n = length(x0)
    x_curr = copy(x0)
    ∇f_curr = ∇f(x_curr)

    B = initial_approximation(inverse, scheme, n)

    d = -solve(B, ∇f_curr)
    α, f_α, ls_success = backtracking(f, x_curr, d, ∇f_curr; α_0=1.0, c=c)
    Δx = α .* d
    x_next = x_curr + Δx
    ∇f_next = ∇f(x_next)
    y = ∇f_next - ∇f_curr
    update!(scheme, B, Δx, y)

    for i = 1:maxiter
        x_curr = copy(x_next)
        ∇f_curr = ∇f(x_curr)
        d = .- solve(B, ∇f_curr)
        α, f_α, ls_success = backtracking(f, x_curr, d, ∇f_curr; α_0=1.0, c=c)
        Δx = α .* d
        x_next = x_curr + Δx
        ∇f_next = ∇f(x_next)
        y = ∇f_next - ∇f_curr

        if norm(∇f_next) < g_tol
            return x_next, ∇f_next
        end
        if show_trace
            println("Objective value (curr): ", f(x_next))
            println("Objective value (prev): ", f(x_curr))
            println("Step size: ", α)
        end
        update!(scheme, B, Δx, y)
    end
    return x_next, ∇f_next
end
