import NearlyNewton: minimize!
function theta(x)
    if x[1] > 0
        return atan(x[2] / x[1]) / (2.0 * pi)
    else
        return (pi + atan(x[2] / x[1])) / (2.0 * pi)
    end
end

function fletcher_powell_fg!(∇f, x)
    theta_x = theta(x)

    if !(∇f==nothing)
        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        end
        ∇f[1] = -2000.0*(x[3]-10.0*theta_x)*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
        ∇f[2] = -2000.0*(x[3]-10.0*theta_x)*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
        ∇f[3] =  200.0*(x[3]-10.0*theta_x) + 2.0*x[3];
    end

    fx = 100.0 * ((x[3] - 10.0 * theta_x)^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

    return fx
end

tp_fletch_powell_fg! = TestProblem(fletcher_powell_fg!, [-1.0, 0.0, 0.0], I, NearlyNewton.OptOptions())
tp_fletch_powell_fg!_alt = TestProblem(fletcher_powell_fg!, [-0.5, 0.0, 0.0], I, NearlyNewton.OptOptions())

minimize!(tp_fletch_powell_fg!, BFGS(InverseApprox()))
minimize!(tp_fletch_powell_fg!, BFGS(DirectApprox()))

minimize!(tp_fletch_powell_fg!, SR1(InverseApprox()))
minimize!(tp_fletch_powell_fg!, SR1(DirectApprox()))

minimize!(tp_fletch_powell_fg!, DFP(InverseApprox()))
minimize!(tp_fletch_powell_fg!, DFP(DirectApprox()))

minimize!(tp_fletch_powell_fg!_alt, BFGS(InverseApprox()))
minimize!(tp_fletch_powell_fg!_alt, BFGS(DirectApprox()))

minimize!(tp_fletch_powell_fg!_alt, SR1(InverseApprox()))
minimize!(tp_fletch_powell_fg!_alt, SR1(DirectApprox()))

minimize!(tp_fletch_powell_fg!_alt, DFP(InverseApprox()))
minimize!(tp_fletch_powell_fg!_alt, DFP(DirectApprox()))
