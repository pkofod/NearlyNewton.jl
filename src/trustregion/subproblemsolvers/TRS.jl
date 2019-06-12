struct TRSolver{T} <: NearlyExactTRSP
    abstol::T
    maxiter::Integer
end
function (ms::TRSolver)(∇f::AbstractVector{T}, H, Δ, p) where T
    x, info = trs(H, ∇f, Δ)
    p .= x[:,1]

    m = dot(∇f, p) + dot(p, H * p)/2
    interior = norm(p, 2) ≤ Δ
    return (p=p, mz=m, interior=interior, λ=info.λ, hard_case=info.hard_case, solved=true)
end
