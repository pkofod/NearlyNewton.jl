abstract type SubProblemSolver end
include("subproblemsolvers/iterative.jl")


function optimize(obj, approach::Tuple{<:Any, <:SubProblemSolver})
    tr_optimize(obj, approach)
end

function tr_optimize(obj, approach)
end
