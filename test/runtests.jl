using NearlyNewton
using Test
using StaticArrays
using Optim
using LineSearches
using Printf
using LinearAlgebra: norm, I
@testset "NearlyNewton.jl" begin

    function theta(x)
       if x[1] > 0
           return atan(x[2] / x[1]) / (2.0 * pi)
       else
           return (pi + atan(x[2] / x[1])) / (2.0 * pi)
       end
    end
    f(x) = 100.0 * ((x[3] - 10.0 * theta(x))^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

    function f∇f!(∇f, x)
        if !(∇f==nothing)
            if ( x[1]^2 + x[2]^2 == 0 )
                dtdx1 = 0;
                dtdx2 = 0;
            else
                dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
                dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            end
            ∇f[1] = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
            ∇f[2] = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
            ∇f[3] =  200.0*(x[3]-10.0*theta(x)) + 2.0*x[3];
        end

        if ∇f == nothing
            return f(x)
        else
            return f(x), ∇f
        end
    end
    function f∇f(∇f, x)
        if !(∇f == nothing)
            return f∇f!(similar(x), x)
        else
            return f∇f!(∇f, x)
        end
    end
    function f∇fs(∇f, x)
        if !(∇f == nothing)
            if ( x[1]^2 + x[2]^2 == 0 )
                dtdx1 = 0;
                dtdx2 = 0;
            else
                dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
                dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) )
            end

            s1 = -2000.0*(x[3]-10.0*theta(x))*dtdx1 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 )
            s2 = -2000.0*(x[3]-10.0*theta(x))*dtdx2 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 )
            s3 = 200.0*(x[3]-10.0*theta(x)) + 2.0*x[3]
            ∇f = @SVector [s1, s2, s3]
            return f(x), ∇f
        else
            return f(x)
        end
    end

    x0 = [-1.0, 0.0, 0.0]
    xopt = [1.0, 0.0, 0.0]
    println("Starting  from: ", x0)
    println("Targeting     : ", xopt)

    shortname(::NearlyNewton.GradientDescent) = "GD  "
    shortname(::NearlyNewton.InverseApprox) = "(inverse)"
    shortname(::NearlyNewton.DirectApprox) = " (direct)"

    function printed_minimize(f∇f, x0, method, approx, B0)
        res = minimize(f∇f, x0, method, approx, B0, NearlyNewton.OptOptions())
        print("NN  $(shortname(method)) $(shortname(approx)): ")
        @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    end
    function printed_minimize!(f∇f!, x0, method, approx, B0)
        res = minimize!(f∇f!, x0, method, approx, B0, NearlyNewton.OptOptions())
        print("NN! $(shortname(method)) $(shortname(approx)): ")
        @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    end

    printed_minimize(f∇f, x0, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    printed_minimize(f∇f, x0, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    printed_minimize!(f∇f!, x0, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    printed_minimize!(f∇f!, x0, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)

    res = minimize(f∇f, x0, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I, NearlyNewton.OptOptions())
    @printf("NN  BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I, NearlyNewton.OptOptions())
    @printf("NN! BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, x0, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I, NearlyNewton.OptOptions())
    @printf("NN  BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I, NearlyNewton.OptOptions())
    @printf("NN! BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    # res = optimize(f∇f!, x0, Optim.BFGS(linesearch=BackTracking()))
    # @time optimize(f∇f!, x0, Optim.BFGS(linesearch=BackTracking()))
    # @printf("OT! BFGS (inverse): %2.2e  %2.2e\n", norm(Optim.minimizer(res)-xopt,Inf), Optim.g_residual(res))

    res = minimize(f∇f, x0, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN  DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN! DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, x0, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN  DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN! DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize(f∇f, x0, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN  SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, x0, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN  SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN! SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN! SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    xrand = rand(3)
    println("\nFrom a random point: ", xrand)
    res = minimize(f∇f, xrand, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN  GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN  GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN! GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN! GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize(f∇f, xrand, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN  BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN  BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN! BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN! BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize(f∇f, xrand, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN  DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN  DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN! DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN! DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize(f∇f, xrand, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN  SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN  SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN! SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN! SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])


    x0s = @SVector [-1.0, 0.0, 0.0]
    println("\nStatic array input.")
    res = minimize(f∇fs, x0s, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN  GD(S)   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN  GD(S)    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN  BFGS(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN  BFGS(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN  DFP(S)  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN  DFP(S)   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN  SR1(S)  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN  SR1(S)   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    function himmelblau!(∇f, x)

        if !(∇f == nothing)
            ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
                44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
            ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        end

        f = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

        if ∇f == nothing
            return f
        else
            return f, ∇f
        end
    end

    himmelblau(∇f, x) = himmelblau!(∇f == nothing ? ∇f : similar(x), x)

    himmelblau_gradient(x) = himmelblau_gradient!(similar(x), x)

    println("\nHimmelblau function")
    x0 = [3.0, 1.0]
    res = minimize(himmelblau, x0, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN  GD   (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN  GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN! GD   (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN! GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, x0, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN  BFGS (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN  BFGS  (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN! BFGS (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN! BFGS  (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    #res = optimize(himmelblau, x0, Optim.BFGS(linesearch=BackTracking()))
    #@printf("OT! BFGS (inverse): %2.2e\n", Optim.g_residual(res))

    res = minimize(himmelblau, x0, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, x0, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, x0, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    xrand = rand(2)
    println("\nFrom a random point: ", xrand)

    res = minimize(himmelblau, xrand, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN  GD   (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN  GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, xrand, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN! GD   (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, xrand, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN! GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, xrand, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN  BFGS (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN  BFGS  (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, xrand, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN! BFGS (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau, xrand, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN! BFGS  (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, xrand, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau,  xrand, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau,  xrand, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, xrand, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau,  xrand, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau,  xrand, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    x0s = @SVector [3.0, 1.0]
    println("\nStatic array input.")
    res = minimize(himmelblau, x0s, NearlyNewton.GradientDescent(), NearlyNewton.InverseApprox(), I)
    @printf("NN  GD(S)   (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, NearlyNewton.GradientDescent(), NearlyNewton.DirectApprox(), I)
    @printf("NN  GD(S)    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, x0s, NearlyNewton.BFGS(), NearlyNewton.InverseApprox(), I)
    @printf("NN  BFGS(S) (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, NearlyNewton.BFGS(), NearlyNewton.DirectApprox(), I)
    @printf("NN  BFGS(S)  (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, x0s, NearlyNewton.DFP(), NearlyNewton.InverseApprox(), I)
    @printf("NN  DFP(S)  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, NearlyNewton.DFP(), NearlyNewton.DirectApprox(), I)
    @printf("NN  DFP(S)   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, x0s, NearlyNewton.SR1(), NearlyNewton.InverseApprox(), I)
    @printf("NN  SR1(S)  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, NearlyNewton.SR1(), NearlyNewton.DirectApprox(), I)
    @printf("NN  SR1(S)   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

end

include("geometrytypes.jl")
