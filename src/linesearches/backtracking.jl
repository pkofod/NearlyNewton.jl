function backtracking(f, x::AbstractArray{T}, d, ∇f_x; α_0=T(1.0),
                      ratio=T(0.5), c=T(0.001), max_iter=1000,
                      verbose=false) where T
    if verbose
        println("Entering line search with step size: ", α_0)
        println("Initial value: ", f(x))
    end

    t = -dot(d, ∇f_x)*c
    α, β = α_0, α_0

    iter = 0

    f_α = f(x + α*d) # initial function value
	is_solved = false
    while !is_solved && iter <= max_iter
        iter += 1
        β, α = α, α*ratio # backtrack according to specified ratio
        f_α = f(x + α*d) # update function value
		is_solved = isfinite(f_α) && f(x + α*d) - f(x) >= α*t
        if verbose
            println("α: ", α)
            println("α*t: ", α*t)
            println("Value at α: ", f(x + α*d))
        end
    end

	ls_succes = iter >= max_iter ? false : true

    if verbose
		!ls_success && println("max_iter exceeded in backtracking")
        println("Exiting line search with step size: ", α, value f_α)
    end
    return α, f_α, ls_success
end
