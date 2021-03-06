# The BFGS update
# The inverse can either be updated with
#  H = (I-s*y'/y'*s)*H*(I-y*s'/y'*s)/(s'y)^2 - (H*y*s' + s*y'*H)/s'*y
# or
#  H = H + (s'*y + y'*H*y)*(s*s')/(s'*y)^2 - (B*y*s' + s*y'*H)/(s'*y)
#

struct BFGS{T1} <: QuasiNewton{T1}
   approx::T1
end
# function update!(scheme::BFGS, B::DirectApprox, Δx, y)
#    B.A = B.A + y*y'/dot(Δx, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::BFGS, B::InverseApprox, Δx, y)
#    B.A = (I - Δx*y'/dot(y, Δx))*B.A*(I - y*Δx'/dot(y, Δx)) + Δx*Δx'/dot(y, Δx)
# end
#
function update(H, s, y, scheme::BFGS{<:InverseApprox})
   sy = dot(s, y)
   ρ = inv(sy)

   if isfinite(ρ)
      C = (I - ρ*s*y')
      H = C*H*C' + ρ*s*s'
   end

   H
end
function update(B, s, y, scheme::BFGS{<:DirectApprox})
   ρ = inv(dot(s, y))
   B + ρ*y*y' - B*s*s'*B/(s'*B*s)
end
function update!(H, s, y, scheme::BFGS{<:InverseApprox})
   sy = dot(s, y)
   ρ = inv(sy)

   if isfinite(ρ)
      Hy = H*y
      H .= H .+ ((sy+y'*Hy).*ρ^2)*s*s'
      Hys = Hy*s'
      H .= H .- (Hys + Hys').*ρ
   end
   H
end
# end
# function update!(H, s, y, scheme::BFGS{<:InverseApprox})
#    sy = dot(s, y)
#    ρ = inv(sy)
#
#    if isfinite(ρ)
#       C = (I - ρ.*s*y')
#       H .= C*H*C' + ρ*s*s'
#    end
#    H
# end
function update!(H, s, y, scheme::BFGS{<:DirectApprox})
    ρ = inv(dot(s, y))
    H .+= ρ*y*y' - H*(s*s')*H/(s'*H*s)
end

function update!(A::UniformScaling, s, y, scheme::BFGS{<:InverseApprox})
   update(A, s, y, scheme)
end
function update!(A::UniformScaling, s, y, scheme::BFGS{<:DirectApprox})
   update(A, s, y, scheme)
end
