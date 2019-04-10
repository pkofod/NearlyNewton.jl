struct SR1 <: BroydenFamily  end
function update(scheme::SR1, approx::InverseApprox, A, s, y)
   sAy = s - A*y
   A + sAy*sAy'/dot(sAy, y)
end
function update(scheme::SR1, approx::DirectApprox, B, s, y)
   yBs = y - B*s
   if dot(yBs, s) < 1e-6
      return B
   else
      return B + yBs*yBs'/dot(yBs, s)
   end
end
function update!(scheme::SR1, approx::InverseApprox, A, s, y)
   sAy = s - A*y
   A .+= sAy*sAy'/dot(sAy, y)
end
function update!(scheme::SR1, approx::DirectApprox, B, s, y)
   yBs = y - B*s
   if dot(yBs, s) > 1e-6
      B .+= yBs*yBs'/dot(yBs, s)
   end
   B
end
function update!(scheme::SR1, approx::InverseApprox, A::UniformScaling, s, y)
   update(scheme, approx, A, s, y)
end
function update!(scheme::SR1, approx::DirectApprox, A::UniformScaling, s, y)
   update(scheme, approx, A, s, y)
end

function find_direction(scheme::SR1, ::DirectApprox, A::AbstractArray, ∇f)
   -(A\∇f)
end
function find_direction!(d::AbstractArray, scheme::SR1, ::DirectApprox, B::AbstractArray, ∇f)
   d .= -(B\∇f)
   d
end
