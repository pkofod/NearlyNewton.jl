struct DFP{T1} <: QuasiNewton{T1}
   approx::T1
end
# function update!(scheme::DFP, B::InverseApprox, s, y)
#    B.A = B.A + s*s'/dot(s, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::DFP, B::DirectApprox, s, y)
#    B.A = (I - y*s'/dot(y, s))*B.A*(I - s*y'/dot(y, s)) + y*y'/dot(y, s)
# end


function update(A, s, y, scheme::DFP{<:InverseApprox})
   first_update(A, y, s)
end
function update(A, s, y, scheme::DFP{<:DirectApprox})
   second_update(A, y, s)
end
function update!(A, s, y, scheme::DFP{<:InverseApprox})
   first_update!(A, y, s)
end
function update!(A, s, y, scheme::DFP{<:DirectApprox})
   second_update!(A, y, s)
end
update!(A::UniformScaling, s, y , scheme::DFP{<:InverseApprox}) = update(A, s, y, scheme)
update!(A::UniformScaling, s, y , scheme::DFP{<:DirectApprox}) = update(A, s, y, scheme)
