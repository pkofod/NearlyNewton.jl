[MORESORENSEN] Computing a trust region step
"""
    is_hardcase(H_eigv, qg)

Returns a tuple of a Bool, `hardcase` and an integer, `λidx`. `hardcase` is true
if the sub-problem is the "hard case", that is the case where the smallest
eigenvalue is negative. `λidx` is the index of the first eigenvalue not equal
to the smallest eigenvalue. `H_eigv` holds the eigenvalues of H sorted low to high,
`qg` is a vector of the inner products between the eigenvectors and the gradient.
"""
function is_hardcase(H_eig, qg)
    # If the solution to the trust region sub-problem is on the boundary of the
    # trust region, {w : ||w|| ≤ Δk}, then the solution is usually found by
    # finding a λ ≥ 0 such that ||(B + λI)⁻¹g|| = Δ and x'(B + λI)x > 0, x≠0.
    # However, in the hard case our strategy for iteratively finding such an
    # λ breaks down, because ||p(λ)|| does not go to ∞ as λ→-λⱼ for j such that
    # qₜ'g ≠ 0 (see section 4.3 in [N&W] for more details). Remember, that B
    # does *not* have to be pos. def. in the trust region based methods! 


    # Get number of eigen values
    λnum = length(qg)
    # The hard case requires a negativ smallest eigenvalue.
    λmin = first(H_eig.values) # eigenvalues are sorted
    if λmin >= 0
        return false, -1
    end

    # Assume hard case and verify
    hard_case = true
    λidx = 1

    hard_case_check_done = false
    while !hard_case_check_done
        if abs(H_eigv[1] - H_eigv[λidx]) > 1e-10
            break
        else
            if abs(qg[λidx]) > 1e-10
                hard_case_check_done = true
                hard_case = false
            end
            λidx += 1
            if λidx > λnum
                break
            end
        end
    end

    hard_case, λidx
end

# Equation 4.38 in N&W (2006)
function calc_p!(lambda::T, min_i, n, qg, H_eig, p) where T
    fill!( p, zero(T) )
    for i = min_i:n
        p[:] -= qg[i] / (H_eig.values[i] + lambda) * H_eig.vectors[:, i]
    end
    return nothing
end

# Choose a point in the trust region for the next step using
# the interative (nearly exact) method of section 4.3 of N&W (2006).
# This is appropriate for Hessians that you factorize quickly.
#
# Args:
#  gr: The gradient
#  H:  The Hessian
#  delta:  The trust region size, ||s|| <= delta
#  s: Memory allocated for the step size, updated in place
#  tolerance: The convergence tolerance for root finding
#  max_iters: The maximum number of root finding iterations
#
# Returns:
#  m - The numeric value of the quadratic minimization.
#  interior - A boolean indicating whether the solution was interior
#  lambda - The chosen regularizing quantity
#  hard_case - Whether or not it was a "hard case" as described by N&W (2006)
#  reached_solution - Whether or not a solution was reached (as opposed to
#      terminating early due to max_iters)
function solve_tr_subproblem!(gr,
                              H,
                              delta,
                              s;
                              tolerance=1e-10,
                              max_iters=5)
    T = eltype(gr)
    n = length(gr)
    delta_sq = delta^2

    @assert n == length(s)
    @assert (n, n) == size(H)
    @assert max_iters >= 1

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    H_eig = eigen(Symmetric(H))
    min_H_ev, max_H_ev = H_eig.values[1], H_eig.values[n]
    H_ridged = copy(H)

    # Cache the inner products between the eigenvectors and the gradient.
    qg = H_eig.vectors' * gr

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    interior = true
    hard_case = false
    reached_solution = true

    # Unconstrained solution
    if min_H_ev >= 1e-8
        calc_p!(zero(T), 1, n, qg, H_eig, s)
    end

    if min_H_ev >= 1e-8 && sum(abs2, s) <= delta_sq
        # No shrinkage is necessary: -(H \ gr) is the minimizer
        interior = true
        reached_solution = true
        lambda = zero(T)
    else
        interior = false

        # The hard case is when the gradient is orthogonal to all
        # eigenvectors associated with the lowest eigenvalue.
        hard_case_candidate, min_i = check_hard_case_candidate(H_eig, qg)

        # Solutions smaller than this lower bound on lambda are not allowed:
        # they don't ridge H enough to make H_ridge PSD.
        lambda_lb = -min_H_ev + max(1e-8, 1e-8 * (max_H_ev - min_H_ev))
        lambda = lambda_lb

        hard_case = false
        if hard_case_candidate
            # The "hard case". lambda is taken to be -min_H_ev and we only need
            # to find a multiple of an orthogonal eigenvector that lands the
            # iterate on the boundary.

            # Formula 4.45 in N&W (2006)
            calc_p!(lambda, min_i, n, qg, H_eig, s)
            p_lambda2 = sum(abs2, s)
            if p_lambda2 > delta_sq
                # Then we can simply solve using root finding.
            else
                hard_case = true
                reached_solution = true

                tau = sqrt(delta_sq - p_lambda2)

                # I don't think it matters which eigenvector we pick so take
                # the first.
                calc_p!(lambda, min_i, n, qg, H_eig, s)
                s[:] = -s + tau * H_eig.vectors[:, 1]
            end
        end

        if !hard_case
            # Algorithim 4.3 of N&W (2006), with s insted of p_l for consistency
            # with Optim.jl

            reached_solution = false
            for iter in 1:max_iters
                lambda_previous = lambda

                for i=1:n
                    H_ridged[i, i] = H[i, i] + lambda
                end

                R = cholesky(Hermitian(H_ridged)).U
                s[:] = -R \ (R' \ gr)
                q_l = R' \ s
                norm2_s = dot(s, s)
                lambda_update = norm2_s * (sqrt(norm2_s) - delta) / (delta * dot(q_l, q_l))
                lambda += lambda_update

                # Check that lambda is not less than lambda_lb, and if so, go
                # half the way to lambda_lb.
                if lambda < lambda_lb
                    lambda = 0.5 * (lambda_previous - lambda_lb) + lambda_lb
                end

                if abs(lambda - lambda_previous) < tolerance
                    reached_solution = true
                    break
                end
            end
        end
    end

    m = dot(gr, s) + 0.5 * dot(s, H * s)

    return m, interior, lambda, hard_case, reached_solution
end
