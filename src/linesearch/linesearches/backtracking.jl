struct BackTracking{T1, T2} <: LineSearch
    ratio::T1
	c::T1
	maxiter::T2
	verbose::Bool
end
BackTracking(; ratio=0.5, c=1e-4, maxiter=50, verbose=false) = BackTracking(ratio, c, maxiter, verbose)
function (ls::BackTracking)(f∇f::T, d, x, f_0, ∇f_0, α_0) where T
	ratio, c, maxiter, verbose = ls.ratio, ls.c, ls.maxiter, ls.verbose


    if verbose
        println("Entering line search with step size: ", α_0)
        println("Initial value: ", f_0)
        println("Value at first step: ", f∇f(nothing, x+α_0*d))
    end

	m = dot(d, ∇f_0)
    t = -c*m

    α, β = α_0, α_0

    iter = 0

    f_α = f∇f(nothing, x + α*d) # initial function value

	is_solved = isfinite(f_α) && f_α <= f_0 + c*α*t
    while !is_solved && iter <= maxiter
        iter += 1
        β, α = α, α*ratio # backtrack according to specified ratio
        f_α = f∇f(nothing, x + α*d) # update function value
		is_solved = isfinite(f_α) && f_α <= f_0 + c*α*t
    end

	ls_success = iter >= maxiter ? false : true
	#
    # if verbose
	# 	!ls_success && println("maxiter exceeded in backtracking")
    #     println("Exiting line search with step size: ", α)
    #     println("Exiting line search with value: ", f_α)
    # end
    return α, f_α, ls_success
end
function backtracking!(objective, x, d, f_0, ∇f_0, α_0, opt)
	ratio, c, maxiter, verbose = opt.ratio, opt.c, opt.maxiter, opt.verbose


    # if verbose
    #     println("Entering line search with step size: ", α_0)
    #     println("Initial value: ", f(x))
    #     println("Value at first step: ", f(x+α_0*d))
    # end

	m = dot(d, ∇f_0)
    t = -c*m

    α, β = α_0, α_0

    iter = 0

    f_α = objective(nothing, nothing, x + α*d) # initial function value
	is_solved = false
    while !is_solved && iter <= maxiter
        iter += 1
        β, α = α, α*ratio # backtrack according to specified ratio
        f_α = objective(nothing, nothing, x + α*d) # update function value
		is_solved = isfinite(f_α) && f_0 - f_α >= α*t
    end

	ls_success = iter >= maxiter ? false : true
	#
    # if verbose
	# 	!ls_success && println("maxiter exceeded in backtracking")
    #     println("Exiting line search with step size: ", α)
    #     println("Exiting line search with value: ", f_α)
    # end
    return α, f_α, ls_success
end
