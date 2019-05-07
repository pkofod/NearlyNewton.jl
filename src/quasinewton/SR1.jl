# make QuasiNewton{SR1, approx} callable and then pass that as the
# "update" parameter
# the minimize can take a single SR1 input as -> QuasiNewton{SR1, Approx}
struct SR1{T1} <: QuasiNewton{T1}
   approx::T1
end
function update(H, s, y, scheme::SR1{<:InverseApprox})
   sHy = s - H*y
   H + sHy*sHy'/dot(sHy, y)
end
function update(B, s, y, scheme::SR1{<:DirectApprox})
   yBs = y - B*s
   d_yBs_s = inv(dot(yBs, s))
   if d_yBs_s > 1e6
      B += d_yBs_s*yBs*yBs'
   end
   B
end
function update!(H, s, y, scheme::SR1{<:InverseApprox})
   sHy = s - H*y
   H .+= sHy*sHy'/dot(sHy, y)
end
function update!(B, s, y, scheme::SR1{<:DirectApprox})
   yBs = y - B*s
   d_yBs_s = inv(dot(yBs, s))
   if d_yBs_s > 1e6
      B .+= d_yBs_s*yBs*yBs'
   end
   B
end
function update!(A::UniformScaling, s, y, scheme::SR1{<:InverseApprox})
   update(A, s, y, scheme)
end
function update!(A::UniformScaling, s, y, scheme::SR1{<:DirectApprox})
   update(A, s, y, scheme)
end

function find_direction(A::AbstractArray, scheme::SR1, ::DirectApprox, ∇f)
   -(A\∇f)
end
function find_direction!(d::AbstractArray, B::AbstractArray, scheme::SR1, ::DirectApprox, ∇f)
   d .= -(B\∇f)
   d
end
