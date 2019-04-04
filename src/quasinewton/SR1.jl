struct SR1 <: BroydenFamily  end
function update(scheme::SR1, ::InverseApprox, A, Δx, y)
   A + (Δx - A*y)*(Δx - A*y)'/dot(Δx - A*y, y)
end
function update(scheme::SR1, B::DirectApprox, A, Δx, y)
   A + (y - A*Δx)*(y - A*Δx)'/dot(y - A*Δx, Δx)
end
