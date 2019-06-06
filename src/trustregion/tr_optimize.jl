# [SOREN] TRUST REGION MODIFICATION OF NEWTON'S METHOD
# [N&W] Numerical optimization
# [Yuan] A review of trust region algorithms for optimization
abstract type SubProblemSolver end
include("subproblemsolvers/iterative.jl")


function optimize(obj, approach::Tuple{<:Any, <:SubProblemSolver})
    tr_optimize(obj, approach)
end

function tr_optimize(obj, approach)
    # first evaluation
    f, ∇f = objective(true, x0)

    # first iteration
    x, f, ∇f, B, is_converged = iterate(B0, ∇f, f, x0, approach, objective, options)

end

struct SubProblemResult
    d
    m
    interior
    lambda
    hard_case
    solved
end

function iterate(B, ∇fx, fx, x, approach, objective, options)
    scheme, subproblemsolver = approach
    # Chosing a parameter > 0 might be preferable here. See p. 4 of Yuans survey
    t0 = 0.1
    t2 = T(1)/4
    t3 = t2 # could differ!
    t4 = T(1)/2
    λ34 = 0.5
    γ = T(2) # gamma for grow
    λγ = 0.5 # distance along growing interval ∈ (0, 1]
    Δmax = T(10)^5 # restrict the largest step
    σ = T(1)/4
    spr = subproblemsolver()

    mz = spr.mz
    # Grab the model value, m. If m is zero, the solution, z, does not improve
    # the model value over x. If the model is not converged, but the optimal
    # step is inside the trust region and gives a zero improvement in the objec-
    # tive value, we may conclude that "something" is wrong. We might be at a
    # ridge (positive-indefinite case) for example, or the scaling of the model
    # is such that we cannot satisfy ||∇f|| < tol.
    if abs(mz) < eps(T)
        # set flag to check for problems
    end

    d = spr.d

    z = @. x + d

    fz, ∇fz = objective(true, z)

    # Δf is often called ared or Ared for actual reduction. I prefer "change in"
    # f, or Delta f.
    Δf = fx - fz

    # Calculate the ratio of actual improvement over predicted improvement.
    R = -Δf/mz

    # We accept all steps larger than t0 ∈ [0, 1/4). See p. 415 of [SOREN] and
    # p.79 as well as  Theorem 4.5 and 4.6 of [N&W]. A t0 = 0 might cycle,
    # see p. 4 of [YUAN].
    if !(t0 <= R)
        z = x
        fz = fx
        ∇fz = ∇fx

        # If you reject an interior solution, make sure that the next
        # delta is smaller than the current step. Otherwise you waste
        # steps reducing Δk by constant factors while each solution
        # will be the same.
        Δkp1 = σ * norm(d)

        accept = false
    else
        # While we accept also the steps in the case that t0 <= Δf < t2, we do not
        # trust it too much. As a result, we restrict the trust region radius. The
        # new trust region radius should be set to a radius Δkp1 ∈ [t3*||d||, t4*Δk].
        # We use the number λ34 ∈ [0, 1] to move along the interval. [N&W] sets
        # λ34 _= 1 and t4 = 1/4, see Algorithm 4.1 on p. 69.
        if R < t2
            Δkp1 = λ34*norm(d, 2)*t3 + (1-λ34)*Δk*t4
        else
            Δkp1 = min(λγ*Δk+(1-λ)*Δk*γ, Δmax)
        end

        accept = true
    end

   end
   return z, fz, ∇z, accept
end
