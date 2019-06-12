function fourth(∇f, x)
    fourth(nothing, ∇f, x)
end
function fourth(∇²f, ∇f, x)
    if !(∇²f == nothing)
        ∇²f = 12x^2
    end
    if !(∇f == nothing)
        ∇f = 4x^3
    end

    fx = x^4
    if ∇f == nothing && ∇²f == nothing
        return fx
    elseif ∇²f == nothing
        return fx, ∇f
    else
        return fx, ∇f, ∇²f
    end
end

minimize(fourth, 4.0, BFGS(DirectApprox()))
