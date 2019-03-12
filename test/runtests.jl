using NearlyNewton
using Test
using StaticArrays
using Optim
using LineSearches
using Printf
using LinearAlgebra: norm
@testset "NearlyNewton.jl" begin

    function f(x)
        function theta(x)
           if x[1] > 0
               return atan(x[2] / x[1]) / (2.0 * pi)
           else
               return (pi + atan(x[2] / x[1])) / (2.0 * pi)
           end
        end

        return 100.0 * ((x[3] - 10.0 * theta(x))^2 +
           (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2
    end

    function g!(storage::Vector, x::Vector)
       function theta(x::Vector)
           if x[1] > 0
               return atan(x[2] / x[1]) / (2.0 * pi)
           else
               return (pi + atan(x[2] / x[1])) / (2.0 * pi)
           end
       end

       if ( x[1]^2 + x[2]^2 == 0 )
           dtdx1 = 0;
           dtdx2 = 0;
       else
           dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
           dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
       end

       storage[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
           200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
       storage[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
           200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
       storage[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
       storage
    end

    g(x) = g!(copy(x), x)

    function gs(x)
        function theta(x)
            if x[1] > 0
                return atan(x[2] / x[1]) / (2.0 * pi)
            else
                return (pi + atan(x[2] / x[1])) / (2.0 * pi)
            end
        end

        if ( x[1]^2 + x[2]^2 == 0 )
            dtdx1 = 0;
            dtdx2 = 0;
        else
            dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
        end

        s1 = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
        s2 = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
            200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
        s3 = 200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
        @SVector [s1, s2, s3]
    end

    x0 = [0.1, 0.1, 0.1]
    xopt = [1.0, 0.0, 0.0]
    println("Starting  from: ", x0)
    println("Targeting from: ", xopt)
    res = minimize(f, g, NearlyNewton.BFGS(), x0; inverse = true)
    @printf("NN  BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f, g, NearlyNewton.BFGS(), x0; inverse = false)
    @printf("NN  BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize!(f, g!, NearlyNewton.BFGS(), x0; inverse = true)
    @printf("NN! BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f, g!, NearlyNewton.BFGS(), x0; inverse = false)
    @printf("NN! BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = optimize(f, g!, x0, Optim.BFGS(linesearch=BackTracking()))
    @printf("OT! BFGS (inverse): %2.2e  %2.2e\n", norm(Optim.minimizer(res)-xopt,Inf), Optim.g_residual(res))

    res = minimize(f, g, NearlyNewton.DFP(), x0; inverse = true)
    @printf("NN  DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f, g, NearlyNewton.DFP(), x0; inverse = false)
    @printf("NN  DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize!(f, g!, NearlyNewton.DFP(), x0; inverse = true)
    @printf("NN! DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f, g!, NearlyNewton.DFP(), x0; inverse = false)
    @printf("NN! DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize(f, g, NearlyNewton.SR1(), x0; inverse = true)
    @printf("NN  SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f, g, NearlyNewton.SR1(), x0; inverse = false)
    @printf("NN  SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize!(f, g!, NearlyNewton.SR1(), x0; inverse = true)
    @printf("NN! SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f, g!, NearlyNewton.SR1(), x0; inverse = false)
    @printf("NN! SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize(f, g, NearlyNewton.GradientDescent(), x0; inverse = true)
    @printf("NN  GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f, g, NearlyNewton.GradientDescent(), x0; inverse = false)
    @printf("NN  GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize!(f, g!, NearlyNewton.GradientDescent(), x0; inverse = true)
    @printf("NN! GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f, g!, NearlyNewton.GradientDescent(), x0; inverse = false)
    @printf("NN! GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    xrand = rand(3)
    println("From a random point: ", xrand)
    res = minimize(f, g, NearlyNewton.GradientDescent(), xrand; inverse = true)
    @printf("NN  GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f, g, NearlyNewton.GradientDescent(), xrand; inverse = false)
    @printf("NN  GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f, g!, NearlyNewton.GradientDescent(), xrand; inverse = true)
    @printf("NN! GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f, g!, NearlyNewton.GradientDescent(), xrand; inverse = false)
    @printf("NN! GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])


    x0s = @SVector [0.1, 0.1, 0.1]
    res = minimize(f, gs, NearlyNewton.GradientDescent(), x0s; inverse = true)
    @printf("NN  GD(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f, gs, NearlyNewton.GradientDescent(), x0s; inverse = false)
    @printf("NN  GD(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    function himmelblau(x)
        return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
    end

    function himmelblau_gradient!(storage, x)
        storage[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        storage[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        storage
    end

    function himmelblau_gradients(x)
        s1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
            44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
        s2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
            4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        @SVector [s1, s2]
    end

    himmelblau_gradient(x) = himmelblau_gradient!(similar(x), x)

    x0 = [3.0, 1.0]
    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.BFGS(), x0; inverse = true)
    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.BFGS(), x0; inverse = false)

    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.DFP(), x0; inverse = true)
    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.DFP(), x0; inverse = false)

    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.SR1(), x0; inverse = true)
    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.SR1(), x0; inverse = false)

    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.GradientDescent(), x0; inverse = true)
    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.GradientDescent(), x0; inverse = false)

    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.GradientDescent(), rand(2); inverse = true)
    @show minimize(himmelblau, himmelblau_gradient, NearlyNewton.GradientDescent(), rand(2); inverse = false)

    x0s = @SVector [3.0, 1.0]
    @show minimize(himmelblau, himmelblau_gradients, NearlyNewton.GradientDescent(), x0s; inverse = true)
    @show minimize(himmelblau, himmelblau_gradients, NearlyNewton.GradientDescent(), x0s; inverse = false)

    x0 = [3.0, 1.0]
    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.BFGS(), x0; inverse = true)
    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.BFGS(), x0; inverse = false)

    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.DFP(), x0; inverse = true)
    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.DFP(), x0; inverse = false)

    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.SR1(), x0; inverse = true)
    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.SR1(), x0; inverse = false)

    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.GradientDescent(), x0; inverse = true)
    @show minimize!(himmelblau, himmelblau_gradient!, NearlyNewton.GradientDescent(), x0; inverse = false)

    x0s = @SVector [3.0, 1.0]
    @show minimize(himmelblau, himmelblau_gradients, NearlyNewton.GradientDescent(), x0s; inverse = true)
    @show minimize(himmelblau, himmelblau_gradients, NearlyNewton.GradientDescent(), x0s; inverse = false)


end
