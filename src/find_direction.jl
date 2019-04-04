# solve the linear system Bx = ∇f for x
find_direction(B::AbstractArray, ∇f) = -B\∇f
find_direction(B::UniformScaling, ∇f) = -∇f
find_direction(::InverseApprox, ::UniformScaling, ∇f) = -∇f
find_direction(::DirectApprox, ::UniformScaling, ∇f) = -∇f
# solve the linear system Bx = ∇f for x inplace
# we may want to accept a factorized B here instead of B itself.
# This way we can simply update the factorization outside.
# do factorization another place for the possibility to do smart things
function find_direction!(d::AbstractArray, B::UniformScaling, ∇f)
   @. d = -∇f
end
function find_direction!(d::AbstractArray, B::AbstractArray, ∇f)
   @. d = -∇f
   ldiv!(factorize(B), d)
   d
end
# If the Hessian approximation is on the actual Hessian, and not its inverse,
# then we will just pass on Happrox.A through the solve-interface above
find_direction(::DirectApprox, A::AbstractArray, ∇f) = find_direction(A, ∇f)
find_direction!(d, ::DirectApprox, A::AbstractArray, ∇f) = find_direction!(d, A, ∇f)
function find_direction!(d::AbstractArray, approx, ::UniformScaling, ∇f)
   @.d = -∇f
   d
end
# Construct an interface to solving for step-sizes if the Hessian approximation
# is held over the inverse not the actual matrix. Then the solve amounts to an
# * or lmul!
find_direction(::InverseApprox, A::AbstractArray, ∇f) = -A*∇f
function find_direction!(d, ::InverseApprox, A::AbstractArray, ∇f)
   mul!(d, A, ∇f)
   rmul!(d, -1)
   d
end
