using GeometryTypes
function f(F, G, x)
    if !(F == nothing)
        fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    end
    if !(G == nothing)
        G1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G2 = 200.0 * (x[2] - x[1]^2)
        gx = Point(G1,G2)
    end
    if !(F == nothing)
        if !(G == nothing)
            return fx, gx
        end
        return fx
    else
        return gx
    end
end

minimize(f, Point(3.0,3.0), NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I, NearlyNewton.OptOptions())
