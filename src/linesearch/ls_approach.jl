abstract type LineSearch end
include("linesearches/backtracking.jl")
include("ls_optimize_inplace.jl")
include("ls_optimize.jl")

function prepare_variables(objective, approach::Tuple{<:Any, <:LineSearch}, x0, ∇fz, B)
    z = x0
    x = copy(z)

    if isa(B, Nothing)  # didn't provide a B
        if isa(first(approach), GradientDescent)
            # We don't need to maintain a dense matrix for Gradient Descent
            B = I
        else
            # Construct a matrix on the correct form and of the correct type
            # with the content of I_{n,n}
            B = I + abs.(0*x*x')
        end
    end
    # first evaluation
    if isa(first(approach), Newton)
        fz, ∇fz, B = objective(B, ∇fz, x)
    else

        fz, ∇fz = objective(∇fz, x)
    end
    fx = copy(fz)
    ∇fx = copy(∇fz)
    return x, fx, ∇fx, z, fz, ∇fz, B
end

function converged(z, ∇fz, g_tol)
    g_converged = norm(∇fz) < g_tol
    return g_converged || any(isnan.(z))
end
