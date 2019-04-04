struct BFGS <: BroydenFamily  end
# function update!(scheme::BFGS, B::DirectApprox, Δx, y)
#    B.A = B.A + y*y'/dot(Δx, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::BFGS, B::InverseApprox, Δx, y)
#    B.A = (I - Δx*y'/dot(y, Δx))*B.A*(I - y*Δx'/dot(y, Δx)) + Δx*Δx'/dot(y, Δx)
# end
#
function update(scheme::BFGS, ::DirectApprox, A, Δx, y)
   first_update(A, Δx, y)
end
function update(scheme::BFGS, ::InverseApprox, A, Δx, y)
   second_update(A, Δx, y)
end
