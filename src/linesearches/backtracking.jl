function backtracking(f, x::AbstractArray{T}, d, ∇f_x; α_0=T(1.0),
                      ratio=T(0.5), c=T(0.001), ls_max_iter=50,
                      verbose=false) where T
    if verbose
        println("Entering line search with step size: ", α_0)
        println("Initial value: ", f(x))
        println("Value at first step: ", f(x+α_0*d))
    end

	m = dot(d, ∇f_x)
    t = -c*m

    α, β = α_0, α_0

    iter = 0

    f_α = f(x + α*d) # initial function value
	is_solved = false
    while !is_solved && iter <= ls_max_iter
        iter += 1
        β, α = α, α*ratio # backtrack according to specified ratio
        f_α = f(x + α*d) # update function value
		is_solved = isfinite(f_α) && f(x) - f(x + α*d) >= α*t
    end

	ls_success = iter >= ls_max_iter ? false : true

    if verbose
		!ls_success && println("ls_max_iter exceeded in backtracking")
        println("Exiting line search with step size: ", α)
        println("Exiting line search with value: ", f_α)
    end
    return α, f_α, ls_success
end
