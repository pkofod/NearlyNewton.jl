# [SOREN] TRUST REGION MODIFICATION OF NEWTON'S METHOD
# [N&W] Numerical optimization
# [Yuan] A review of trust region algorithms for optimization
abstract type TRSPSolver end
abstract type NearlyExactTRSP <: TRSPSolver end

include("subproblemsolvers/NWI.jl")

function minimize!(obj, x0, approach::Tuple{<:Any, <:TRSPSolver}, B0=nothing, options=OptOptions())
    tr_minimize!(obj, x0, approach, B0, options)
end

function tr_minimize!(obj, x0, approach, B0, options)
    T = eltype(x0)

    scheme = first(approach)
    if isa(B0, Nothing)
        if isa(scheme, GradientDescent)
            B0 = I # we don't add 0.0*x*x' here for gradient descent
        else
            B0 = I + T(0)*x0*x0'
        end
    end
    # first evaluation
    x = copy(x0)
    z = copy(x0)
    ∇fx = copy(x0)
    ∇fz = copy(x0)

    # first evaluation
    if isa(scheme, Newton)
        fz, ∇fz, B0 = obj(B0, ∇fz, x0)
    else
        fz, ∇fz = obj(∇fz, x0)
    end

    Δk = T(10.0)

    z, fz, ∇fz, Bz, Δkp1, accept, is_converged = iterate!(x, ∇fx, z, fz, ∇fz, B0, Δk, approach, obj, options)

    iter = 0
    is_converged = false
    while iter <= options.max_iter && !is_converged
        iter += 1
        z, fz, ∇fz, Bz, Δkp1, accept, is_converged = iterate!(x, ∇fx, z, fz, ∇fz, Bz, Δkp1, approach, obj, options)
        @show Δkp1
        @show Bz
        @show ∇fz
    end
    return z, fz, ∇fz, iter
end


function iterate!(x::AbstractArray{T}, ∇fx, z, fz, ∇fz, Bx, Δk, approach, objective, options; scale=false) where T
    scheme, subproblemsolver = approach
    fx = fz
    copyto!(∇fx, ∇fz)
    copyto!(x, z)
    # Chosing a parameter > 0 might be preferable here. See p. 4 of Yuans survey
    # We want to avoid cycles, but we also need something that takes very small
    # steps when convergence is hard to achieve.
    α = 0.15 # acceptance ratio
    t2 = T(1)/4
    t3 = t2 # could differ!
    t4 = T(1)/2
    λ34 = T(0)/2
    γ = T(2.5) # gamma for grow
    λγ = T(0)/2 # distance along growing interval ∈ (0, 1]
    Δmax = T(10)^5 # restrict the largest step
    σ = T(1)/4
    p = copy(x)
    spr = subproblemsolver(∇fx, Bx, Δk, p; abstol=1e-10, maxiter=50)
    Δm = -spr.mz
    # Grab the model value, m. If m is zero, the solution, z, does not improve
    # the model value over x. If the model is not converged, but the optimal
    # step is inside the trust region and gives a zero improvement in the objec-
    # tive value, we may conclude that "something" is wrong. We might be at a
    # ridge (positive-indefinite case) for example, or the scaling of the model
    # is such that we cannot satisfy ||∇f|| < tol.
    if abs(spr.mz) < eps(T)
        # set flag to check for problems
    end

    # add Retract
    z .= x + p

    # Update before acceptance, to keep adding information about the hessian
    # even when the step is not "good" enough.
    fz, ∇fz, Bz = update_qn(objective, z, spr.p, spr.p, ∇fx, ∇fz, Bx, scheme, scale)

    # Δf is often called ared or Ared for actual reduction. I prefer "change in"
    # f, or Delta f.
    Δf = fx - fz

    # Calculate the ratio of actual improvement over predicted improvement.
    R = Δf/Δm

    # We accept all steps larger than α ∈ [0, 1/4). See p. 415 of [SOREN] and
    # p.79 as well as  Theorem 4.5 and 4.6 of [N&W]. A α = 0 might cycle,
    # see p. 4 of [YUAN].
    if !(α <= R)
        z .= x
        fz = fx
        ∇fz .= ∇fx

        if spr.interior
            # If you reject an interior solution, make sure that the next
            # delta is smaller than the current step. Otherwise you waste
            # steps reducing Δk by constant factors while each solution
            # will be the same.
            Δkp1 = σ * norm(p)
        else
            Δkp1 = λ34*norm(p, 2)*t3 + (1-λ34)*Δk*t4
        end
        accept = false
    else
        # While we accept also the steps in the case that α <= Δf < t2, we do not
        # trust it too much. As a result, we restrict the trust region radius. The
        # new trust region radius should be set to a radius Δkp1 ∈ [t3*||d||, t4*Δk].
        # We use the number λ34 ∈ [0, 1] to move along the interval. [N&W] sets
        # λ34 _= 1 and t4 = 1/4, see Algorithm 4.1 on p. 69.
        if R < t2
            Δkp1 = λ34*norm(p, 2)*t3 + (1-λ34)*Δk*t4
        else
            Δkp1 = min(λγ*Δk+(1-λγ)*Δk*γ, Δmax)
        end
        accept = true
    end
    is_converged = converged(z, ∇fz, options.g_tol)

    return z, fz, ∇fz, Bz, Δkp1, accept, is_converged
end
