using NearlyNewton
using Test
using StaticArrays
#using Optim
#using LineSearches
using Printf
using LinearAlgebra: norm, I

include("noalloc.jl")

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

        fx = f(x)
        return ∇f==nothing ? fx : (fx, ∇f)
    end

    function f∇f(∇f, x)
        if !(∇f == nothing)
            gx = similar(x)
            return f∇f!(gx, x)
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

    function printed_minimize(f∇f, x0, method, B0=nothing)
        res = minimize(f∇f, x0, method, B0, NearlyNewton.OptOptions())
        print("NN  $(shortname(method)) $(shortname(method.approx)): ")
        @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    end
    function printed_minimize!(f∇f!, x0, method, B0=nothing)
        res = minimize!(f∇f!, x0, method, B0, NearlyNewton.OptOptions())
        print("NN! $(shortname(method)) $(shortname(method.approx)): ")
        @printf("%2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    end

    printed_minimize(f∇f, x0, GradientDescent(InverseApprox()))
    printed_minimize!(f∇f!, copy(x0), GradientDescent(InverseApprox()))
    res = minimize(f∇fs, x0s, GradientDescent(InverseApprox()))
    @printf("NN  GD(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

    printed_minimize(f∇f, x0, GradientDescent(DirectApprox()))
    printed_minimize!(f∇f!, copy(x0), GradientDescent(DirectApprox()))
    res = minimize(f∇fs, x0s, GradientDescent(DirectApprox()))
    @printf("NN  GD(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    println()

    res = minimize(f∇f, x0, BFGS(InverseApprox()), nothing, NearlyNewton.OptOptions())
    @test res[4] == 30
    @printf("NN  BFGS    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(x0), (BFGS(InverseApprox()), BackTracking()), nothing, NearlyNewton.OptOptions())
    @test res[4] == 30
    @printf("NN! BFGS    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, x0s, BFGS(InverseApprox()))
    @test res[4] == 30
    @printf("NN  BFGS(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

    res = minimize(f∇f, x0, BFGS(DirectApprox()), nothing, NearlyNewton.OptOptions())
    @printf("NN  BFGS    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(x0), BFGS(DirectApprox()), nothing, NearlyNewton.OptOptions())
    @printf("NN! BFGS    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, x0s, BFGS(DirectApprox()))
    @printf("NN  BFGS(S) (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    println()
    # res = optimize(f∇f!, copy(x0), Optim.BFGS(linesearch=BackTracking()))
    # @time optimize(f∇f!, copy(x0), Optim.BFGS(linesearch=BackTracking()))
    # @printf("OT! BFGS (inverse): %2.2e  %2.2e\n", norm(Optim.minimizer(res)-xopt,Inf), Optim.g_residual(res))

    res = minimize(f∇f, x0, DFP(InverseApprox()))
    @printf("NN  DFP    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(x0), DFP(InverseApprox()))
    @printf("NN! DFP    (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, x0s, DFP(InverseApprox()))
    @printf("NN  DFP(S) (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇f, x0, DFP(DirectApprox()))
    @printf("NN  DFP    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(x0), DFP(DirectApprox()))
    @printf("NN! DFP    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, x0s, DFP(DirectApprox()))
    @printf("NN  DFP(S) (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    println()

    res = minimize(f∇f, x0, SR1(InverseApprox()))
    @printf("NN  SR1     (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(x0), SR1(InverseApprox()))
    @printf("NN! SR1     (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, x0s, SR1(InverseApprox()))
    @printf("NN  SR1(S)  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇f, x0, SR1(DirectApprox()))
    @printf("NN  SR1     (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(x0), SR1(DirectApprox()))
    @printf("NN! SR1     (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, x0s, SR1(DirectApprox()))
    @printf("NN  SR1(S)  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

    xrand = rand(3)
    xrands = SVector{3}(xrand)
    println("\nFrom a random point: ", xrand)
    res = minimize(f∇f, xrand, GradientDescent(InverseApprox()))
    @printf("NN  GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), GradientDescent(InverseApprox()))
    @printf("NN! GD   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇f, xrand, GradientDescent(DirectApprox()))
    @printf("NN  GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), GradientDescent(DirectApprox()))
    @printf("NN! GD    (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    println()
    res = minimize(f∇f, xrand, BFGS(InverseApprox()))
    @printf("NN  BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), (BFGS(InverseApprox()), BackTracking()))
    @printf("NN! BFGS (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇f, xrand, BFGS(DirectApprox()))
    @printf("NN  BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), BFGS(DirectApprox()))
    @printf("NN! BFGS  (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    println()

    res = minimize(f∇f, xrand, DFP(InverseApprox()))
    @printf("NN  DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), DFP(InverseApprox()))
    @printf("NN! DFP  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇f, xrand, DFP(DirectApprox()))
    @printf("NN  DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), DFP(DirectApprox()))
    @printf("NN! DFP   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    println()

    res = minimize(f∇f, xrand, SR1(InverseApprox()))
    @printf("NN  SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), SR1(InverseApprox()))
    @printf("NN! SR1  (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, xrand, SR1(InverseApprox()))
    @printf("NN  SR1(S)   (inverse): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇f, xrand, SR1(DirectApprox()))
    @printf("NN  SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize!(f∇f!, copy(xrand), SR1(DirectApprox()))
    @printf("NN! SR1   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])
    res = minimize(f∇fs, xrand, SR1(DirectApprox()))
    @printf("NN  SR1(S)   (direct): %2.2e  %2.2e  %d\n", norm(res[1]-xopt,Inf), norm(res[3], Inf), res[4])

    println()


    function himmelblau!(∇f, x)
        if !(∇f == nothing)
            ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
                44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
            ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        end

        fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
        return ∇f == nothing ? fx : (fx, ∇f)
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

        return himmelblau!(g, x)
    end


    println("\nHimmelblau function")
    x0 = [3.0, 1.0]
    x0s = @SVector([3.0, 1.0])
    res = minimize(himmelblau, x0, GradientDescent(InverseApprox()))
    @printf("NN  GD    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), GradientDescent(InverseApprox()))
    @printf("NN! GD    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblaus, x0s, GradientDescent(InverseApprox()))
    @printf("NN  GD(S) (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    println()
    res = minimize(himmelblau, x0, GradientDescent(DirectApprox()))
    @printf("NN  GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), GradientDescent(DirectApprox()))
    @printf("NN! GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblaus, x0s, GradientDescent(DirectApprox()))
    @printf("NN  GD(S) (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    println()

    res = minimize(himmelblau, x0, BFGS(InverseApprox()))
    @printf("NN  BFGS    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), BFGS(InverseApprox()))
    @printf("NN! BFGS    (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0s, BFGS(InverseApprox()))
    @printf("NN  BFGS(S) (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    println()
    res = minimize(himmelblau, x0, BFGS(DirectApprox()))
    @printf("NN  BFGS    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), BFGS(DirectApprox()))
    @printf("NN! BFGS    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0s, BFGS(DirectApprox()))
    @printf("NN  BFGS(S) (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    println()

    res = minimize(himmelblau, x0, DFP(InverseApprox()))
    @printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), DFP(InverseApprox()))
    @printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0s, DFP(InverseApprox()))
    @printf("NN  DFP(S)  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0, DFP(DirectApprox()))
    println()
    @printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), DFP(DirectApprox()))
    @printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0s, DFP(DirectApprox()))
    @printf("NN  DFP(S)   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    println()

    res = minimize(himmelblau, x0, SR1(InverseApprox()))
    @printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), SR1(InverseApprox()))
    @printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0s, SR1(InverseApprox()))
    @printf("NN  SR1(S)  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0, SR1(DirectApprox()))
    @printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(x0), SR1(DirectApprox()))
    @printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, x0s, SR1(DirectApprox()))
    @printf("NN  SR1(S)   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    println()
    xrand = rand(2)
    println("\nFrom a random point: ", xrand)

    res = minimize(himmelblau, xrand, GradientDescent(InverseApprox()))
    @printf("NN  GD   (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(xrand), GradientDescent(InverseApprox()))
    @printf("NN! GD   (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, xrand, GradientDescent(DirectApprox()))
    @printf("NN  GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(xrand), GradientDescent(DirectApprox()))
    @printf("NN! GD    (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

    res = minimize(himmelblau, xrand, BFGS(InverseApprox()))
    @printf("NN   BFGS (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(xrand), BFGS(InverseApprox()))
    @printf("NN!  BFGS (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, xrand, BFGS(DirectApprox()))
    @printf("NN(S) BFGS (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, xrand, BFGS(DirectApprox()))
    @printf("NN  BFGS  (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!, copy(xrand), BFGS(DirectApprox()))
    @printf("NN! BFGS  (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

    res = minimize(himmelblau, xrand, DFP(InverseApprox()))
    @printf("NN  DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!,  copy(xrand), DFP(InverseApprox()))
    @printf("NN! DFP  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, xrand, DFP(DirectApprox()))
    @printf("NN  DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!,  copy(xrand), DFP(DirectApprox()))
    @printf("NN! DFP   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

    res = minimize(himmelblau, xrand, SR1(InverseApprox()))
    @printf("NN  SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!,  copy(xrand), SR1(InverseApprox()))
    @printf("NN! SR1  (inverse): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize(himmelblau, xrand, SR1(DirectApprox()))
    @printf("NN  SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])
    res = minimize!(himmelblau!,  copy(xrand), SR1(DirectApprox()))
    @printf("NN! SR1   (direct): %2.2e  %d\n", norm(res[3], Inf), res[4])

end

include("geometrytypes.jl")
include("newton.jl")
