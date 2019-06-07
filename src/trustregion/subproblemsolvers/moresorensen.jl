# [MORESORENSEN] Computing a trust region step
# This file contains the Newton Trust Region Subproblem solver described in
# [MORESORENSEN] and [N&W]. In the latter it's called the "iterative" solution
# method, and the method is a near-exact solution method. It's popular and ef-
# ficient even in cases that are problematic (known as the "hard case" after
# [MORESORENSEN]) that are not guaranteed to be solved (at all or quickly)
# by older methods. The method is appropriate if factorization is fast and
# feasible.

struct MoreSorensen <: NearlyExactTRSP

end

"""
    is_maybe_hard_case(QΛQ, Qt∇f)

Returns a tuple of a Bool, `hardcase` and an integer, `λidx`. `hardcase` is true
if the sub-problem is the "hard case", that is the case where the smallest
eigenvalue is negative. `λidx` is the index of the first eigenvalue not equal
to the smallest eigenvalue. `QΛQ.values` holds the eigenvalues of H sorted low
to high, `Qt∇f` is a vector of the inner products between the eigenvectors and the
gradient.
"""
function is_maybe_hard_case(QΛQ, Qt∇f::AbstractVector{T}) where T
    # If the solution to the trust region sub-problem is on the boundary of the
    # trust region, {w : ||w|| ≤ Δk}, then the solution is usually found by
    # finding a λ ≥ 0 such that ||(B + λI)⁻¹g|| = Δ and x'(B + λI)x > 0, x≠0.
    # However, in the hard case our strategy for iteratively finding such an
    # λ breaks down, because ||p(λ)|| does not go to ∞ as λ→-λⱼ for j such that
    # qₜ'g ≠ 0 (see section 4.3 in [N&W] for more details). Remember, that B
    # does *not* have to be pos. def. in the trust region based methods!

    # Get the eigenvalues
    Λ = QΛQ.values

    # Get number of eigen values
    λnum = length(Λ)

    # The hard case requires a negativ smallest eigenvalue.
    λmin = first(Λ) # eigenvalues are sorted
    if λmin >= T(0)
        return false, 1
    end

    # Assume hard case and verify
    λidx = 1
    for (Qt∇f_j, λ_j in zip(Qt∇f, Λ)
        if abs(λmin - λ_j) > sqrt(eps(T))
            hard_case = true
            break
        else
            if abs(Qt∇f_j) > sqrt(eps(T))
                hard_case = false
                break
            end
        end
        λidx += 1
    end

    hard_case, λidx
end

# Equation 4.38 in N&W (2006)
calc_p!(p, Qt∇f, QΛQ, λ) = calc_p!(p, Qt∇f, QΛQ, λ, 1)

# Equation 4.45 in N&W (2006) since we allow for first_j > 1
function calc_p!(p, Qt∇f, QΛQ, λ::T, first_j) where T
    # Reset search direction to 0
    fill!(p, T(0))

    # Unpack eigenvalues and eigenvectors
    Λ = QΛQ.values
    Q = QΛQ.vectors

    for j = first_j:length(Λ)
        κ = Qt∇f[j] / (Λ[j] + λ)
        @. p = p - κ*Q[:, j]
    end
    p
end

"""
    solve_tr_subproblem!(∇f, H, Δ, s; abstol, maxiter)
Args:
    ∇f: The gradient
    H:  The Hessian
    Δ:  The trust region size, ||s|| <= Δ
    s: Memory allocated for the step size, updated in place
    abstol: The convergence abstol for root finding
    maxiter: The maximum number of root finding iterations

Returns:
    m - The numeric value of the quadratic minimization.
    interior - A boolean indicating whether the solution was interior
    lambda - The chosen regularizing quantity
    hard_case - Whether or not it was a "hard case" as described by N&W (2006)
    solved - Whether or not a solution was reached (as opposed to
      terminating early due to maxiter)
"""
function (ms::MoreSorensen)!(∇f::AbstractVector{T},
                              H, Δ, p;
                              abstol=1e-10, maxiter=5) where T

    n = length(∇f)

    # Note that currently the eigenvalues are only sorted if H is perfectly
    # symmetric.  (Julia issue #17093)
    QΛQ = eigen(Symmetric(H))
    Q, Λ = QΛQ.vectors, QΛQ.values

    λmin, λmax = Λ[1], Λ[n]
    H_ridged = copy(H)

    # Cache the inner products between the eigenvectors and the gradient.
    Qt∇f = Q' * ∇f

    # These values describe the outcome of the subproblem.  They will be
    # set below and returned at the end.
    solved = true

    # Potentially an unconstrained/interior solution. The smallest eigenvalue is
    # positive, so the Newton step, pN, is fine unless norm(pN, 2) > Δ.
    if λmin >= sqrt(eps(T))
        λ = T(0)
        p = calc_p!(p, Qt∇f, QΛQ, λ)

        if norm(p, 2) ≤ Δ
            # No shrinkage is necessary: -(H \ ∇f) is the minimizer
            interior = true
            solved = true
            hard_case = false

            m = dot(∇f, p) + 0.5 * dot(p, H * p)

            return (p=p, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved)
        end
    end

    # Set interior flag
    interior = false

    # The hard case is when the gradient is orthogonal to all
    # eigenvectors associated with the lowest eigenvalue.
    maybe_hard_case, first_j = is_maybe_hard_case(QΛQ, Qt∇f)

    # Solutions smaller than this lower bound on lambda are not allowed:
    # they don't ridge H enough to make H_ridge PSD.
    λ_lb = -λmin + max(sqrt(eps(T)), sqrt(eps(T)) * (λmax - λmin))
    λ = λ_lb

    # Verify that it is actually the hard case situation by calculating the
    # step with λ = λmin (it's currently λ_lb, verify that that is correct).
    if maybe_hard_case
        # The "hard case".
        # λ is taken to be -λmin and we only need to find a multiple of an
        # orthogonal eigenvector that lands the iterate on the boundary.

        # The old p is discarded, and replaced with one that takes into account
        # the first j such that λj ≠ λmin. Formula 4.45 in N&W (2006)
        pλ = calc_p!(p, λ, first_j, n, Qt∇f, QΛQ)

        # Check if the choice of λ leads to a solution inside the trust region.
        # If it does, then we construct the "hard case solution".
        if norm(pλ, 2) ≤ Δ
            hard_case = true
            solved = true

            tau = sqrt(Δ^2 - norm(pλ, 2)^2)

            @. p = -pλ + tau * Q[:, 1]

            m = dot(∇f, p) + 0.5 * dot(p, H * p)

            return (p=p, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved)
        end
        # If this is reached, we cannot be in the hard case after all, and we
        # can use Newton's method to find a p such that norm(p, 2) = Δ.
    end

    # Algorithim 4.3 of N&W (2006), with s insted of p_l for consistency
    # with Optim.jl

    solved = false
    for iter in 1:maxiter
        λ_previous = λ

        for i = 1:n
            @inplace H_ridged[i, i] = H[i, i] + λ
        end

        R = cholesky(Hermitian(H_ridged)).U
        p .= -R \ (R' \ ∇f)
        q_l = R' \ p

        p_norm = norm(p, 2)
        λ_update = p_norm^2 * (p_norm - Δ) / (Δ * dot(q_l, q_l))
        λ += λ_update

        # Check that λ is not less than λ_lb, and if so, go
        # half the way to λ_lb.
        if λ < λ_lb
            λ = 0.5 * (λ_previous - λ_lb) + λ_lb
        end

        if abs(λ - λ_previous) ≤ abstol
            solved = true
            break
        end
    end

    m = dot(∇f, p) + 0.5 * dot(p, H * p)

    return (p=p, mz=m, interior=interior, λ=λ, hard_case=hard_case, solved=solved)
end
