struct BFGS <: BroydenFamily  end
# function update!(scheme::BFGS, B::DirectApprox, Δx, y)
#    B.A = B.A + y*y'/dot(Δx, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::BFGS, B::InverseApprox, Δx, y)
#    B.A = (I - Δx*y'/dot(y, Δx))*B.A*(I - y*Δx'/dot(y, Δx)) + Δx*Δx'/dot(y, Δx)
# end
#
function update(scheme::BFGS, ::DirectApprox, B, s, y)
   if dot(s, y) > 1e-2
      first_update(B, s, y)
   end
   B
end
function update(scheme::BFGS, ::InverseApprox, H, s, y)
   second_update(H, s, y)
end
function update!(scheme::BFGS, ::DirectApprox, A, s, y)
   first_update!(A, s, y)
end
function update!(scheme::BFGS, ::InverseApprox, A, s, y)
   second_update!(A, s, y)
end
function update!(scheme::BFGS, approx::DirectApprox, A::UniformScaling, s, y)
   update(scheme, approx, A, s, y)
end
function update!(scheme::BFGS, approx::InverseApprox, A::UniformScaling, s, y)
   update(scheme, approx, A, s, y)
end
