struct DFP{T1} <: QuasiNewton{T1}
   approx::T1
end
# function update!(scheme::DFP, B::InverseApprox, s, y)
#    B.A = B.A + s*s'/dot(s, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::DFP, B::DirectApprox, s, y)
#    B.A = (I - y*s'/dot(y, s))*B.A*(I - s*y'/dot(y, s)) + y*y'/dot(y, s)
# end


function update(H, s, y, scheme::DFP{<:InverseApprox})
    ρ = inv(dot(s, y))
    H + ρ*s*s' - H*(y*y')*H/(y'*H*y)
end
function update(B, s, y, scheme::DFP{<:DirectApprox})
    ρ = inv(dot(s, y))

    if ρ > 1e6
        C = (I - ρ*y*s')
        B = C*B*C' + ρ*y*y'
    end
    B
end
function update!(H, s, y, scheme::DFP{<:InverseApprox})
    ρ = inv(dot(s, y))
    H .+= ρ*s*s' - H*(y*y')*H/(y'*H*y)
end
function update!(B, s, y, scheme::DFP{<:DirectApprox})
    ρ = inv(dot(s, y))
    # so right now, we just skip the update if dsy is zero
    # but we might do something else here
    if ρ > 1e6
        C = (I - ρ*y*s')
        B .= C*B*C' + ρ*y*y'
    end
    B
end
update!(A::UniformScaling, s, y , scheme::DFP{<:InverseApprox}) = update(A, s, y, scheme)
update!(A::UniformScaling, s, y , scheme::DFP{<:DirectApprox}) = update(A, s, y, scheme)
