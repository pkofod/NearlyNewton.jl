module NearlyNewton

# We use often use the LinearAlgebra functions dot and norm for operations rela-
# ted to assessing angles between vectors, size of vectors and so on. I is mainly
# used to get gradient descent for free out of Newton's method.
using LinearAlgebra: dot, I, norm, mul!, factorize, ldiv!, rmul!

# Include the actual functions that expose the functionality in this package.
include("minimize.jl")
export minimize, minimize!
include("nlsolve.jl")
export nlsolve, nlsolve!

# Include line search functionality; currently  limited to backtracking
include("linesearches/backtracking.jl")
export backtracking

# Export algos
export BFGS, SR1, DFP, GradientDescent


struct GradientDescent end
struct GradientDescentHessianApproximation end
# might want to clean this up
function update!(scheme::GradientDescent, rest...) end

# solve the linear system Bx = ∇f for x
find_direction(B, ∇f) = -B\∇f
find_direction(B::GradientDescentHessianApproximation, ∇f) = -∇f
# solve the linear system Bx = ∇f for x inplace
# we may want to accept a factorized B here instead of B itself.
# This way we can simply update the factorization outside.
# do factorization another place for the possibility to do smart things
function find_direction!(d, B, ∇f)
   @. d = -∇f
   ldiv!(factorize(B), d)
   d
end
# If the Hessian approximation is on the actual Hessian, and not its inverse,
# then we will just pass on Happrox.A through the solve-interface above
mutable struct HessianApproximation{TB}
   A::TB
end
find_direction(B::HessianApproximation, ∇f) = find_direction(B.A, ∇f)
find_direction!(d, B::HessianApproximation, ∇f) = find_direction!(d, B.A, ∇f)
function find_direction!(d, B::GradientDescentHessianApproximation, ∇f)
   @.d = -∇f
   d
end
# Construct an interface to solving for step-sizes if the Hessian approximation
# is held over the inverse not the actual matrix. Then the solve amounts to an
# * or lmul!
mutable struct InverseHessianApproximation{TH}
   A::TH
end
find_direction(B::InverseHessianApproximation, ∇f) = -B.A*∇f
function find_direction!(d, B::InverseHessianApproximation, ∇f)
   mul!(d, B.A, ∇f)
   rmul!(d, -1)
   d
end
# Just some initial approximation to the (inverse) Hessian.
function initial_approximation(inverse, scheme, n)
    if typeof(scheme) <: GradientDescent
        return GradientDescentHessianApproximation()
    end
    if inverse
        A = InverseHessianApproximation(Matrix{Float64}(I, n, n))
    else
        A = HessianApproximation(Matrix{Float64}(I, n, n))
    end
    return A
end

# A NearlyNewtonMethod method has the following:
# *  A type <: NearlyNewtonMethod
# *  A method for `update!(scheme::YourType, B)` for either HessianApproximation.
#    InverseHessianApproximation or both that updates the current approximation
#    to the (inverse) Hessian
# *

abstract type NearlyNewtonMethod end

# Could be called ActualNewton...
struct Newton <: NearlyNewtonMethod end

mutable struct NthNewton <: NearlyNewtonMethod
   counter::Int
   Nth::Int
end
NthNewton(Nth) = NthNewton(Nth-1, Nth)
function update!(scheme::NthNewton, B::HessianApproximation, ∇²f, x)
   scheme.counter += 1
   if scheme.count % scheme.Nth == 0
      B.A = ∇²f
      scheme.counter = 0
   end
   B.A
end

abstract type BroydenFamily <: NearlyNewtonMethod end
struct BFGS <: BroydenFamily  end
# function update!(scheme::BFGS, B::HessianApproximation, Δx, y)
#    B.A = B.A + y*y'/dot(Δx, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::BFGS, B::InverseHessianApproximation, Δx, y)
#    B.A = (I - Δx*y'/dot(y, Δx))*B.A*(I - y*Δx'/dot(y, Δx)) + Δx*Δx'/dot(y, Δx)
# end
#
struct DFP <: BroydenFamily  end
# function update!(scheme::DFP, B::InverseHessianApproximation, Δx, y)
#    B.A = B.A + Δx*Δx'/dot(Δx, y) - B.A*y*y'*B.A/(y'*B.A*y)
# end
# function update!(scheme::DFP, B::HessianApproximation, Δx, y)
#    B.A = (I - y*Δx'/dot(y, Δx))*B.A*(I - Δx*y'/dot(y, Δx)) + y*y'/dot(y, Δx)
# end

struct SR1 <: BroydenFamily  end
function update!(scheme::SR1, B::InverseHessianApproximation, Δx, y)
   B.A = B.A + (Δx - B.A*y)*(Δx - B.A*y)'/dot(Δx - B.A*y, y)
end
function update!(scheme::SR1, B::HessianApproximation, Δx, y)
   B.A = B.A + (y - B.A*Δx)*(y - B.A*Δx)'/dot(y - B.A*Δx, Δx)
end

function update!(scheme::BFGS, B::HessianApproximation, Δx, y)
   B.A = first_update!(B.A, Δx, y)
end
function update!(scheme::DFP, B::InverseHessianApproximation, Δx, y)
   B.A = first_update!(B.A, y, Δx)
end
function update!(scheme::BFGS, B::InverseHessianApproximation, Δx, y)
   B.A = second_update!(B.A, Δx, y)
end
function update!(scheme::DFP, B::HessianApproximation, Δx, y)
   B.A = second_update!(B.A, y, Δx)
end
"""
First update is phi = 0, so BFGS for hessian
"""
function first_update!(X, p, q)
    X + q*q'/dot(p, q) - X*q*q'*X/(q'*X*q)
end
function second_update!(X, p, q)
    (I - p*q'/dot(q, p))*X*(I - q*p'/dot(q, p)) + p*p'/dot(q, p)
end

struct NearlyNewtonUpdater{SchemeType, BType}
   scheme::SchemeType
   B::BType
end

end # module
