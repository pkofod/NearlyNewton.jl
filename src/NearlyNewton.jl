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
# then we will just pass on Happrox.B through the solve-interface above
mutable struct HessianApproximation{TB}
   B::TB
end
find_direction(B::HessianApproximation, ∇f) = find_direction(B.B, ∇f)
find_direction!(d, B::HessianApproximation, ∇f) = find_direction!(d, B.B, ∇f)
function find_direction!(d, B::GradientDescentHessianApproximation, ∇f)
   @.d = -∇f
   d
end
# Construct an interface to solving for step-sizes if the Hessian approximation
# is held over the inverse not the actual matrix. Then the solve amounts to an
# * or lmul!
mutable struct InverseHessianApproximation{TH}
   H::TH
end
find_direction(B::InverseHessianApproximation, ∇f) = -B.H*∇f
function find_direction!(d, B::InverseHessianApproximation, ∇f)
   mul!(d, B.H, ∇f)
   rmul!(d, -1)
   d
end
# Just some initial approximation to the (inverse) Hessian.
function initial_approximation(inverse, scheme, n)
    if typeof(scheme) <: GradientDescent
        return GradientDescentHessianApproximation()
    end
    if inverse
        B = InverseHessianApproximation(Matrix{Float64}(I, n, n))
    else
        B = HessianApproximation(Matrix{Float64}(I, n, n))
    end
    return B
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
      B.B = ∇²f
      scheme.counter = 0
   end
   B.B
end

abstract type BroydenFamily <: NearlyNewtonMethod end
struct BFGS <: BroydenFamily  end
function update!(scheme::BFGS, B::HessianApproximation, Δx, y)
   B.B = B.B + y*y'/dot(Δx, y) - B.B*y*y'*B.B/(y'*B.B*y)
end
function update!(scheme::BFGS, B::InverseHessianApproximation, Δx, y)
   B.H = (I - Δx*y'/dot(y, Δx))*B.H*(I - y*Δx'/dot(y, Δx)) + Δx*Δx'/dot(y, Δx)
end

struct DFP <: BroydenFamily  end
function update!(scheme::DFP, B::InverseHessianApproximation, Δx, y)
   B.H = B.H + Δx*Δx'/dot(Δx, y) - B.H*y*y'*B.H/(y'*B.H*y)
end
function update!(scheme::DFP, B::HessianApproximation, Δx, y)
   B.B = (I - y*Δx'/dot(y, Δx))*B.B*(I - Δx*y'/dot(y, Δx)) + y*y'/dot(y, Δx)
end

struct SR1 <: BroydenFamily  end
function update!(scheme::SR1, B::InverseHessianApproximation, Δx, y)
   B.H = B.H + (Δx - B.H*y)*(Δx - B.H*y)'/dot(Δx - B.H*y, y)
end
function update!(scheme::SR1, B::HessianApproximation, Δx, y)
   B.B = B.B + (y - B.B*Δx)*(y - B.B*Δx)'/dot(y - B.B*Δx, Δx)
end



end # module
