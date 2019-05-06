using GeometryTypes
function f(G, x)
    fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

    if !(G == nothing)
        G1 = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
        G2 = 200.0 * (x[2] - x[1]^2)
        gx = Point(G1,G2)

        return fx, gx
    else
        return fx
    end
end

minimize(f, Point(3.0,3.0), NearlyNewton.GradientDescent(InverseApprox()), I, NearlyNewton.OptOptions())
minimize(f, Point(3.0,3.0), NearlyNewton.BFGS(InverseApprox()), I, NearlyNewton.OptOptions())
minimize(f, Point(3.0,3.0), NearlyNewton.DFP(InverseApprox()), I, NearlyNewton.OptOptions())
minimize(f, Point(3.0,3.0), NearlyNewton.SR1(InverseApprox()), I, NearlyNewton.OptOptions())

minimize(f, Point(3.0,3.0), NearlyNewton.GradientDescent(DirectApprox()), I, NearlyNewton.OptOptions())
minimize(f, Point(3.0,3.0), NearlyNewton.BFGS(DirectApprox()), I, NearlyNewton.OptOptions())
minimize(f, Point(3.0,3.0), NearlyNewton.DFP(DirectApprox()), I, NearlyNewton.OptOptions())
minimize(f, Point(3.0,3.0), NearlyNewton.SR1(DirectApprox()), I, NearlyNewton.OptOptions())
