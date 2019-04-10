module NearlyNewton

# We use often use the LinearAlgebra functions dot and norm for operations rela-
# ted to assessing angles between vectors, size of vectors and so on. I is mainly
# used to get gradient descent for free out of Newton's method.
using LinearAlgebra: dot, I, norm, mul!, cholesky, ldiv!, rmul!, UniformScaling, Symmetric, factorize

# Include the actual functions that expose the functionality in this package.
include("hessian_object.jl")

include("minimize.jl")
export minimize, minimize!, OptOptions
include("nlsolve.jl")
export nlsolve, nlsolve!

# Include line search functionality; currently  limited to backtracking
include("linesearches/backtracking.jl")
export backtracking


abstract type NearlyNewtonMethod end
abstract type BroydenFamily <: NearlyNewtonMethod end
include("./find_direction.jl")
include("./quasinewton/SR1.jl")
include("./quasinewton/formulae.jl")
include("./quasinewton/gradientdescent.jl")
include("./quasinewton/BFGS.jl")
include("./quasinewton/DFP.jl")
# Export algos
export BFGS, SR1, DFP, GradientDescent

struct NearlyNewtonUpdater{SchemeType, BType}
   scheme::SchemeType
   B::BType
end

end # module
