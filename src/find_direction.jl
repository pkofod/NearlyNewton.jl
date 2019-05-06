find_direction(::UniformScaling, ∇f, scheme) = -∇f
function find_direction(A::AbstractArray, ∇f, scheme)
   if typeof(scheme.approx) <: DirectApprox
      return -(A\∇f)
   else
      return -A*∇f
   end
end

# solve the linear system Bx = ∇f for x inplace
# we may want to accept a factorized B here instead of B itself.
# This way we can simply update the factorization outside.
# do factorization another place for the possibility to do smart things

function find_direction!(d::AbstractArray, ::UniformScaling, ∇f, scheme)
   @. d = -∇f
   d
end
# danger zone, we don't check properly for conditions when updating so might
# not be pd
# @. d = -∇f
# ldiv!(cholesky(Symmetric(B)), d)
function find_direction!(d::AbstractArray, B::AbstractArray, ∇f, scheme::QuasiNewton{<:DirectApprox})
   d .= -(B\∇f)
   d
end
function find_direction!(d::AbstractArray, A::AbstractArray, ∇f, scheme::QuasiNewton{<:InverseApprox})
   rmul!(mul!(d, A, ∇f), -1)
   d
end
