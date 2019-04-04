struct DFP <: BroydenFamily  end
# function update!(scheme::DFP, B::InverseApprox, Δx, y)
#    B.A = B.A + Δx*Δx'/dot(Δx, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::DFP, B::DirectApprox, Δx, y)
#    B.A = (I - y*Δx'/dot(y, Δx))*B.A*(I - Δx*y'/dot(y, Δx)) + y*y'/dot(y, Δx)
# end


function update(scheme::DFP, ::InverseApprox, A, Δx, y)
   first_update(A, y, Δx)
end
function update(scheme::DFP, ::DirectApprox, A, Δx, y)
   second_update(A, y, Δx)
end
