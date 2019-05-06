# make QuasiNewton{SR1, approx} callable and then pass that as the
# "update" parameter
# the minimize can take a single SR1 input as -> QuasiNewton{SR1, Approx}
struct SR1{T1} <: QuasiNewton{T1}
   approx::T1
end
function update(A, s, y, scheme::SR1{<:InverseApprox})
   sAy = s - A*y
   A + sAy*sAy'/dot(sAy, y)
end
function update(B, s, y, scheme::SR1{<:DirectApprox})
   yBs = y - B*s
   if dot(yBs, s) < 1e-6
      return B
   else
      return B + yBs*yBs'/dot(yBs, s)
   end
end
function update!(A, s, y, scheme::SR1{<:InverseApprox})
   sAy = s - A*y
   A .+= sAy*sAy'/dot(sAy, y)
end
function update!(B, s, y, scheme::SR1{<:DirectApprox})
   yBs = y - B*s
   if dot(yBs, s) > 1e-6
      B .+= yBs*yBs'/dot(yBs, s)
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
