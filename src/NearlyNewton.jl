module NearlyNewton

using StaticArrays
using LinearAlgebra: dot, I, norm

include("nearlynewton.jl")
export nearlynewton, nearlynewton!

include("linesearches/backtracking.jl")
export backtracking


solve(B, ∇f) = B\∇f
solve!(B, d) = ldiv!(factorize(B), d) # do factorization another place for the possibility to do smart things

mutable struct InverseHessianApproximation{TH}
   H::TH
end
solve(B::InverseHessianApproximation, ∇f) = B.H*∇f
solve!(d, B::InverseHessianApproximation, d) = lmul!(B.H, d)

mutable struct HessianApproximation{TB}
    B::TB
end
solve(B::HessianApproximation, ∇f) = B.B\∇f
solve!(B::HessianApproximation, d) = ldiv!(factorize(B.B), d)

# Just some initial approximation to the (inverse) Hessian.
function initial_approximation(inverse, scheme, n)
    if typeof(scheme) <: GradientDescent
        return I
    end
    if inverse
        B = InverseHessianApproximation(Matrix{Float64}(I, n, n))
    else
        B = HessianApproximation(Matrix{Float64}(I, n, n))
    end
    return B
end

# A NearlyNewton method has the following:
# *  A type <: NearlyNewtonMethod
# *  A method for `update!(scheme::YourType, B)` for either HessianApproximation.
#    InverseHessianApproximation or both.
# *
struct Newton end

#
struct BFGS end
function update!(scheme::BFGS, B::HessianApproximation, Δx, y)
   B.B = B.B + y*y'/dot(Δx, y) - B.B*y*y'*B.B/(y'*B.B*y)
end
function update!(scheme::BFGS, B::InverseHessianApproximation, Δx, y)
   B.H = (I - Δx*y'/dot(y, Δx))*B.H*(I - y*Δx'/dot(y, Δx)) + Δx*Δx'/dot(y, Δx)
end
struct DFP end
function update!(scheme::DFP, B::InverseHessianApproximation, Δx, y)
   B.H = B.H + Δx*Δx'/dot(Δx, y) - B.H*y*y'*B.H/(y'*B.H*y)
end
function update!(scheme::DFP, B::HessianApproximation, Δx, y)
   B.B = (I - y*Δx'/dot(y, Δx))*B.B*(I - Δx*y'/dot(y, Δx)) + y*y'/dot(y, Δx)
end
struct SR1 end

function update!(scheme::SR1, B::InverseHessianApproximation, Δx, y)
   B.H = B.H + (Δx - B.H*y)*(Δx - B.H*y)'/dot(Δx - B.H*y, y)
end
function update!(scheme::SR1, B::HessianApproximation, Δx, y)
   B.B = B.B + (y - B.B*Δx)*(y - B.B*Δx)'/dot(y - B.B*Δx, Δx)
end
struct GradientDescent end
function update!(scheme::GradientDescent, B, Δx, y)
end


end # module
