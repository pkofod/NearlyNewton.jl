function find_direction(A, ∇f, scheme::QuasiNewton{<:DirectApprox})
   return -(A\∇f)
end
function find_direction(A, ∇f, scheme::QuasiNewton{<:InverseApprox})
   return -A*∇f
end

function find_direction!(d, B, ∇f, scheme::QuasiNewton{<:DirectApprox})
   d .= -(B\∇f)
   d
end
function find_direction!(d, A, ∇f, scheme::QuasiNewton{<:InverseApprox})
   rmul!(mul!(d, A, ∇f), -1)
   d
end
