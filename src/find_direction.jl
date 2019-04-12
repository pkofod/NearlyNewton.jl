function find_direction(scheme, approx, ::UniformScaling, ∇f)
   -∇f
end
function find_direction(scheme, ::DirectApprox, A::AbstractArray, ∇f)
   -(A\∇f)
end
find_direction(scheme, ::InverseApprox, A::AbstractArray, ∇f) = -A*∇f

# solve the linear system Bx = ∇f for x inplace
# we may want to accept a factorized B here instead of B itself.
# This way we can simply update the factorization outside.
# do factorization another place for the possibility to do smart things

function find_direction!(d::AbstractArray, scheme, approx, ::UniformScaling, ∇f)
   @. d = -∇f
   d
end
function find_direction!(d::AbstractArray, scheme, ::DirectApprox, B::AbstractArray, ∇f)
   # danger zone, we don't check properly for conditions when updating so might
   # not be pd
   d .= -(B\∇f)
   # @. d = -∇f
   # ldiv!(cholesky(Symmetric(B)), d)
   d
end
function find_direction!(d, scheme, ::InverseApprox, A::AbstractArray, ∇f)
   rmul!(mul!(d, A, ∇f), -1)
   d
end
