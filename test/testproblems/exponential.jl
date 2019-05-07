function exponential_fg!(∇f, x)

    if !(∇f==nothing)
        ∇f[1] = -2.0 * (2.0 - x[1]) * exp((2.0 - x[1])^2)
        ∇f[2] = -2.0 * (3.0 - x[2]) * exp((3.0 - x[2])^2)
    end

    fx = exp((2.0 - x[1])^2) + exp((3.0 - x[2])^2)

    if ∇f == nothing
        return fx
    else
        return fx, ∇f
    end
end


function exponential_hessian!(storage::Matrix, x::Vector)
    storage[1, 1] = 2.0 * exp((2.0 - x[1])^2) * (2.0 * x[1]^2 - 8.0 * x[1] + 9)
    storage[1, 2] = 0.0
    storage[2, 1] = 0.0
    storage[2, 2] = 2.0 * exp((3.0 - x[2])^2) * (2.0 * x[2]^2 - 12.0 * x[2] + 19)
end

tp_fletch_powell_fg! = TestProblem(exponential_fg!, [0.0, 0.0], I, NearlyNewton.OptOptions())
tp_fletch_powell_fg!_alt = TestProblem(exponential_fg!, [0.0, 0.0], I, NearlyNewton.OptOptions())
