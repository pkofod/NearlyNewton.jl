using NearlyNewton
using Test
using StaticArrays
#using Optim
#using LineSearches
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

        return f(x)
    end
    function f∇f(∇f, x)
        if !(∇f == nothing)
            gx = similar(x)
            return f∇f!(gx, x), gx
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
    x0s = @SVector [-1.0, 0.0, 0.0]

    println("Starting  from: ", x0)
    println("Targeting     : ", xopt)

    shortname(::GradientDescent) = "GD  "
    shortname(::InverseApprox) = "(inverse)"
    shortname(::NearlyNewton.DirectApprox) = " (direct)"

    function printed_minimize(f∇f, x0, method, B0)
        res = minimize(f∇f, x0, method, B0, NearlyNewton.OptOptions())
        print("NN  $(shortname(method)) $(shortname(method.approx)): ")
        @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    end
    function printed_minimize!(f∇f!, x0, method, B0)
        res = minimize!(f∇f!, x0, method, B0, NearlyNewton.OptOptions())
        print("NN! $(shortname(method)) $(shortname(method.approx)): ")
        @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    end

    printed_minimize(f∇f, x0, GradientDescent(InverseApprox()), I)
    printed_minimize!(f∇f!, x0, GradientDescent(InverseApprox()), I)
    res = minimize(f∇fs, x0s, GradientDescent(InverseApprox()), I)
    @printf("NN  GD(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    printed_minimize(f∇f, x0, GradientDescent(DirectApprox()), I)
    printed_minimize!(f∇f!, x0, GradientDescent(DirectApprox()), I)
    res = minimize(f∇fs, x0s, GradientDescent(DirectApprox()), I)
    @printf("NN  GD(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    println()

    res = minimize(f∇f, x0, BFGS(InverseApprox()), I, NearlyNewton.OptOptions())
    @printf("NN  BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, (BFGS(InverseApprox()), BackTracking()), I, NearlyNewton.OptOptions())
    @printf("NN! BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, BFGS(InverseApprox()), I)
    @printf("NN  BFGS(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    res = minimize(f∇f, x0, BFGS(DirectApprox()), I, NearlyNewton.OptOptions())
    @printf("NN  BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, BFGS(DirectApprox()), I, NearlyNewton.OptOptions())
    @printf("NN! BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, BFGS(DirectApprox()), I)
    @printf("NN  BFGS(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    println()
    # res = optimize(f∇f!, x0, Optim.BFGS(linesearch=BackTracking()))
    # @time optimize(f∇f!, x0, Optim.BFGS(linesearch=BackTracking()))
    # @printf("OT! BFGS (inverse): %2.2e  %2.2e\n", norm(Optim.minimizer(res)-xopt,Inf), Optim.g_residual(res))

    res = minimize(f∇f, x0, DFP(InverseApprox()), I)
    @printf("NN  DFP    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, DFP(InverseApprox()), I)
    @printf("NN! DFP    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, DFP(InverseApprox()), I)
    @printf("NN  DFP(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, x0, DFP(DirectApprox()), I)
    @printf("NN  DFP    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, DFP(DirectApprox()), I)
    @printf("NN! DFP    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, DFP(DirectApprox()), I)
    @printf("NN  DFP(S) (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    println()

    res = minimize(f∇f, x0, SR1(InverseApprox()), I)
    @printf("NN  SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, SR1(InverseApprox()), I)
    @printf("NN! SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, SR1(InverseApprox()), I)
    @printf("NN  SR1(S)  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, x0, SR1(DirectApprox()), I)
    @printf("NN  SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, x0, SR1(DirectApprox()), I)
    @printf("NN! SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, x0s, SR1(DirectApprox()), I)
    @printf("NN  SR1(S)   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    xrand = rand(3)
    xrands = SVector{3}(xrand)
    println("\nFrom a random point: ", xrand)
    res = minimize(f∇f, xrand, GradientDescent(InverseApprox()), I)
    @printf("NN  GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, GradientDescent(InverseApprox()), I)
    @printf("NN! GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, GradientDescent(DirectApprox()), I)
    @printf("NN  GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, GradientDescent(DirectApprox()), I)
    @printf("NN! GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    println()
    res = minimize(f∇f, xrand, BFGS(InverseApprox()), I)
    @printf("NN  BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, (BFGS(InverseApprox()), BackTracking()), I)
    @printf("NN! BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, BFGS(DirectApprox()), I)
    @printf("NN  BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, BFGS(DirectApprox()), I)
    @printf("NN! BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    println()

    res = minimize(f∇f, xrand, DFP(InverseApprox()), I)
    @printf("NN  DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, DFP(InverseApprox()), I)
    @printf("NN! DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, DFP(DirectApprox()), I)
    @printf("NN  DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, DFP(DirectApprox()), I)
    @printf("NN! DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    println()

    res = minimize(f∇f, xrand, SR1(InverseApprox()), I)
    @printf("NN  SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, SR1(InverseApprox()), I)
    @printf("NN! SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, xrand, SR1(InverseApprox()), I)
    @printf("NN  SR1(S)   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇f, xrand, SR1(DirectApprox()), I)
    @printf("NN  SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize!(f∇f!, xrand, SR1(DirectApprox()), I)
    @printf("NN! SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])
    res = minimize(f∇fs, xrand, SR1(DirectApprox()), I)
    @printf("NN  SR1(S)   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[2], Inf), res[3])

    println()


    function himmelblau!(∇f, x)
        f = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
        if !(∇f == nothing)
            ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
                44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
            ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        end
        return f
    end

    function himmelblaus(∇f, x)
        f = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
        if !(∇f == nothing)
            ∇f1 = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
                44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
            ∇f2 = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
            return f, @SVector([∇f1, ∇f2])
        else
            return f
        end
    end

    function himmelblau(∇f, x)
        g = ∇f == nothing ? ∇f : similar(x)
        if !isa(∇f, Nothing)
            fx = himmelblau!(g, x)
            return fx, g
        else
            return himmelblau!(g, x)
        end
    end


    println("\nHimmelblau function")
    x0 = [3.0, 1.0]
    x0s = @SVector([3.0, 1.0])
    res = minimize(himmelblau, x0, GradientDescent(InverseApprox()), I)
    @printf("NN  GD    (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, GradientDescent(InverseApprox()), I)
    @printf("NN! GD    (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblaus, x0s, GradientDescent(InverseApprox()), I)
    @printf("NN  GD(S) (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0, GradientDescent(DirectApprox()), I)
    println()
    @printf("NN  GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, GradientDescent(DirectApprox()), I)
    @printf("NN! GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblaus, x0s, GradientDescent(DirectApprox()), I)
    @printf("NN  GD(S) (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    println()

    res = minimize(himmelblau, x0, BFGS(InverseApprox()), I)
    @printf("NN  BFGS    (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, BFGS(InverseApprox()), I)
    @printf("NN! BFGS    (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, BFGS(InverseApprox()), I)
    @printf("NN  BFGS(S) (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    println()
    res = minimize(himmelblau, x0, BFGS(DirectApprox()), I)
    @printf("NN  BFGS    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, BFGS(DirectApprox()), I)
    @printf("NN! BFGS    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, BFGS(DirectApprox()), I)
    @printf("NN  BFGS(S) (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    println()

    res = minimize(himmelblau, x0, DFP(InverseApprox()), I)
    @printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, DFP(InverseApprox()), I)
    @printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, DFP(InverseApprox()), I)
    @printf("NN  DFP(S)  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0, DFP(DirectApprox()), I)
    println()
    @printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, DFP(DirectApprox()), I)
    @printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, DFP(DirectApprox()), I)
    @printf("NN  DFP(S)   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    println()

    res = minimize(himmelblau, x0, SR1(InverseApprox()), I)
    @printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, SR1(InverseApprox()), I)
    @printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, SR1(InverseApprox()), I)
    @printf("NN  SR1(S)  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0, SR1(DirectApprox()), I)
    @printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, x0, SR1(DirectApprox()), I)
    @printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, x0s, SR1(DirectApprox()), I)
    @printf("NN  SR1(S)   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    println()
    xrand = rand(2)
    println("\nFrom a random point: ", xrand)

    res = minimize(himmelblau, xrand, GradientDescent(InverseApprox()), I)
    @printf("NN  GD   (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, xrand, GradientDescent(InverseApprox()), I)
    @printf("NN! GD   (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, GradientDescent(DirectApprox()), I)
    @printf("NN  GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, xrand, GradientDescent(DirectApprox()), I)
    @printf("NN! GD    (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, xrand, BFGS(InverseApprox()), I)
    @printf("NN  BFGS (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, xrand, BFGS(InverseApprox()), I)
    @printf("NN! BFGS (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, BFGS(DirectApprox()), I)
    @printf("NN  BFGS  (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!, xrand, BFGS(DirectApprox()), I)
    @printf("NN! BFGS  (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, xrand, DFP(InverseApprox()), I)
    @printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!,  xrand, DFP(InverseApprox()), I)
    @printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, DFP(DirectApprox()), I)
    @printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!,  xrand, DFP(DirectApprox()), I)
    @printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

    res = minimize(himmelblau, xrand, SR1(InverseApprox()), I)
    @printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!,  xrand, SR1(InverseApprox()), I)
    @printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize(himmelblau, xrand, SR1(DirectApprox()), I)
    @printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])
    res = minimize!(himmelblau!,  xrand, SR1(DirectApprox()), I)
    @printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[2], Inf), res[3])

end

include("geometrytypes.jl")

@testset "no-alloc static" begin

    function theta(x)
        if x[1] > 0
            return atan(x[2] / x[1]) / (2.0 * pi)
        else
            return (pi + atan(x[2] / x[1])) / (2.0 * pi)
        end
    end

    function fletcher_powell_fg_static(∇f, x)
        theta_x = theta(x)

        if !(∇f==nothing)
            if ( x[1]^2 + x[2]^2 == 0 )
                dtdx1 = 0;
                dtdx2 = 0;
            else
                dtdx1 = - x[2] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
                dtdx2 =   x[1] / ( 2 * pi * ( x[1]^2 + x[2]^2 ) );
            end
            ∇f1 = -2000.0*(x[3]-10.0*theta_x)*dtdx1 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[1]/sqrt( x[1]^2+x[2]^2 );
            ∇f2 = -2000.0*(x[3]-10.0*theta_x)*dtdx2 +
                200.0*(sqrt(x[1]^2+x[2]^2)-1)*x[2]/sqrt( x[1]^2+x[2]^2 );
            ∇f3 =  200.0*(x[3]-10.0*theta_x) + 2.0*x[3];
            ∇f = @SVector[∇f1, ∇f2, ∇f3]
        end

        fx = 100.0 * ((x[3] - 10.0 * theta_x)^2 + (sqrt(x[1]^2 + x[2]^2) - 1.0)^2) + x[3]^2

        if ∇f == nothing
            return fx
        else
            return fx, ∇f
        end
    end

    @allocated minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], BFGS(InverseApprox()), I, NearlyNewton.OptOptions())
    _alloc = @allocated minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], BFGS(InverseApprox()), I, NearlyNewton.OptOptions())
    @test _alloc == 0

    minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], BFGS(DirectApprox()), I, NearlyNewton.OptOptions())
    _alloc = @allocated minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], BFGS(DirectApprox()), I, NearlyNewton.OptOptions())
    @test _alloc == 0

    minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], DFP(InverseApprox()), I, NearlyNewton.OptOptions())
    _alloc = @allocated minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], DFP(InverseApprox()), I, NearlyNewton.OptOptions())
    @test _alloc == 0

    minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], DFP(DirectApprox()), I, NearlyNewton.OptOptions())
    _alloc = @allocated minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], DFP(DirectApprox()), I, NearlyNewton.OptOptions())
    @test _alloc == 0

    minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], SR1(InverseApprox()), I, NearlyNewton.OptOptions())
    _alloc = @allocated minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], SR1(InverseApprox()), I, NearlyNewton.OptOptions())
    @test _alloc == 0

    minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], SR1(DirectApprox()), I, NearlyNewton.OptOptions())
    _alloc = @allocated minimize(fletcher_powell_fg_static, @SVector[-0.5, 0.0, 0.0], SR1(DirectApprox()), I, NearlyNewton.OptOptions())
    @test _alloc == 0

end
