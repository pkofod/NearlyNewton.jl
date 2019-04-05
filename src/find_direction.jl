find_direction(approx, ::UniformScaling, ∇f) = -∇f
function find_direction(::DirectApprox, A::AbstractArray, ∇f)
   -(factorize(A)\∇f)
end
find_direction(::InverseApprox, A::AbstractArray, ∇f) = -A*∇f

# solve the linear system Bx = ∇f for x inplace
# we may want to accept a factorized B here instead of B itself.
# This way we can simply update the factorization outside.
# do factorization another place for the possibility to do smart things

function find_direction!(d::AbstractArray, approx, ::UniformScaling, ∇f)
   @.d = -∇f
   d
end
function find_direction!(d::AbstractArray, ::DirectApprox, B::AbstractArray, ∇f)
   @. d = -∇f
   ldiv!(factorize(B), d)
   @. d
   # @. d = -∇f
   # ldiv!(factorize(B), d)
   # d
end
function find_direction!(d, ::InverseApprox, A::AbstractArray, ∇f)
   mul!(d, A, ∇f)
   rmul!(d, -1)
   d
end
