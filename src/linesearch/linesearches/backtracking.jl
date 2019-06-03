struct BackTracking{T1, T2} <: LineSearch
    ratio::T1
	c::T1
	max_iter::T2
	verbose::Bool
end
BackTracking(; ratio=0.5, c=1e-4, max_iter=100, verbose=true) = BackTracking(ratio, c, max_iter, verbose)
function (ls::BackTracking)(f∇f, x, d, f_0, ∇f_0, α_0)
    opt = LSOptions(ls.ratio, ls.c, ls.max_iter, ls.verbose)
	backtracking(f∇f, x, d, f_0, ∇f_0, α_0, opt)
end

struct LSOptions{T1, T2, T3}
    ratio::T1
	c::T1
	max_iter::T2
	verbose::T3
end

LSOptions(;ratio=0.5, c=1e-4, max_iter=100, verbose=false) = LSOptions(ratio, c, max_iter, verbose)
function backtracking(f∇f, x, d, f_0, ∇f_0, α_0, opt)
	ratio, c, max_iter, verbose = opt.ratio, opt.c, opt.max_iter, opt.verbose


    # if verbose
    #     println("Entering line search with step size: ", α_0)
    #     println("Initial value: ", f(x))
    #     println("Value at first step: ", f(x+α_0*d))
    # end

	m = dot(d, ∇f_0)
    t = -c*m

    α, β = α_0, α_0

    iter = 0

    f_α = f∇f(nothing, x + α*d) # initial function value
	is_solved = isfinite(f_α) && f_α <= f_0 + c*α*t
    while !is_solved && iter <= max_iter
        iter += 1
        β, α = α, α*ratio # backtrack according to specified ratio
        f_α = f∇f(nothing, x + α*d) # update function value
		is_solved = isfinite(f_α) && f_α <= f_0 + c*α*t
    end

	ls_success = iter >= max_iter ? false : true
	#
    # if verbose
	# 	!ls_success && println("max_iter exceeded in backtracking")
    #     println("Exiting line search with step size: ", α)
    #     println("Exiting line search with value: ", f_α)
    # end
    return α, f_α, ls_success
end
function backtracking!(f∇f!, x, d, f_0, ∇f_0, α_0, opt)
	ratio, c, max_iter, verbose = opt.ratio, opt.c, opt.max_iter, opt.verbose


    # if verbose
    #     println("Entering line search with step size: ", α_0)
    #     println("Initial value: ", f(x))
    #     println("Value at first step: ", f(x+α_0*d))
    # end

	m = dot(d, ∇f_0)
    t = -c*m

    α, β = α_0, α_0

    iter = 0

    f_α = f∇f!(nothing, x + α*d) # initial function value
	is_solved = false
    while !is_solved && iter <= max_iter
        iter += 1
        β, α = α, α*ratio # backtrack according to specified ratio
        f_α = f∇f!(nothing, x + α*d) # update function value
		is_solved = isfinite(f_α) && f_0 - f_α >= α*t
    end

	ls_success = iter >= max_iter ? false : true
	#
    # if verbose
	# 	!ls_success && println("max_iter exceeded in backtracking")
    #     println("Exiting line search with step size: ", α)
    #     println("Exiting line search with value: ", f_α)
    # end
    return α, f_α, ls_success
end
