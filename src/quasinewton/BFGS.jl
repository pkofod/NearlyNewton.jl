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
   second_update(H, s, y)
end
function update(B, s, y, scheme::BFGS{<:DirectApprox})
   first_update(B, s, y)
end
function update!(A, s, y, scheme::BFGS{<:InverseApprox})
   second_update!(A, s, y)
end
function update!(A, s, y, scheme::BFGS{<:DirectApprox})
   first_update!(A, s, y)
end

function update!(A::UniformScaling, s, y, scheme::BFGS{<:InverseApprox})
   update(A, s, y, scheme)
end
function update!(A::UniformScaling, s, y, scheme::BFGS{<:DirectApprox})
   update(A, s, y, scheme)
end
