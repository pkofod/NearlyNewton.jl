module NearlyNewton

# We use often use the LinearAlgebra functions dot and norm for operations rela-
# ted to assessing angles between vectors, size of vectors and so on. I is mainly
# used to get gradient descent for free out of Newton's method.
using LinearAlgebra: dot, I, norm, mul!, cholesky, ldiv!, rmul!, UniformScaling, Symmetric, factorize


# make this struct that has scheme and approx
abstract type QuasiNewton{T1} end


struct OptOptions{T1, T2}
    c::T1
    g_tol::T1
    max_iter::T2
    show_trace::Bool
end

OptOptions(; c=1e-4, g_tol=1e-8, max_iter=10^4, show_trace=false) =
OptOptions(c, g_tol, max_iter, show_trace)

# Include the actual functions that expose the functionality in this package.
include("hessian_object.jl")

include("quasinewton/update_qn.jl")
include("minimize.jl")
export minimize, minimize!, OptOptions
export backtracking, BackTracking
export InverseApprox, DirectApprox


include("nlsolve.jl")
export nlsolve, nlsolve!

struct MinimizationProblem{Tobj, Tcon}
    objective::Tobj
    constraints::Tcon
end
struct MaximizationProblem{Tobj, Tcon}
    objective::Tobj
    constraints::Tcon
end
struct Objective{T, InformationOrder, Inplace}
    func::T
end


abstract type ObjectiveOrder end
struct OrderZero <: ObjectiveOrder end
struct OrderOne <: ObjectiveOrder end
struct OrderTwo <: ObjectiveOrder end

function Objective(order=nothing; f=nothing, g=nothing,
                                  fg=nothing, h=nothing,
                                  fgh=nothing, inplace=true)

    if order == OrderZero && !(f==nothing)
        return f
    end
end

include("./find_direction.jl")
include("./quasinewton/SR1.jl")
include("./quasinewton/formulae.jl")
include("./quasinewton/gradientdescent.jl")
include("./quasinewton/BFGS.jl")
include("./quasinewton/DFP.jl")
# Export algos
export BFGS, SR1, DFP, GradientDescent

end # module
