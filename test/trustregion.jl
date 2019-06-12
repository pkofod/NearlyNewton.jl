using NearlyNewton, LinearAlgebra, StaticArrays

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

    res = NearlyNewton.minimize!(f∇f!, x0, (SR1(DirectApprox()), NearlyNewton.NWI()), I+0.0*x0*x0')
    res = NearlyNewton.minimize!(f∇f!, x0, (BFGS(DirectApprox()), NearlyNewton.NWI()), I+0.0*x0*x0')
    res = NearlyNewton.minimize!(f∇f!, x0, (BFGS(DirectApprox()), BackTracking()), I+0.0*x0*x0')
















    function himmelblau_gradient!(x::Vector, gradient::Vector)
               gradient[1] = 4 * x[1] * (x[1]^2 + x[2] - 11) + 2 * (x[1] + x[2]^2 - 7)
               gradient[2] = 2 * (x[1]^2 + x[2] - 11) + 4 * x[2] * (x[1] + x[2]^2 - 7)
           end

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

    function himmelblau!(∇²f, ∇f, x)
        if !(∇²f == nothing)
            ∇²f[1, 1] = 12.0 * x[1]^2 + 4.0 * x[2] - 42.0
            ∇²f[1, 2] = 4.0 * x[1] + 4.0 * x[2]
            ∇²f[2, 1] = 4.0 * x[1] + 4.0 * x[2]
            ∇²f[2, 2] = 12.0 * x[2]^2 + 4.0 * x[1] - 26.0
        end
        if !(∇f == nothing)
            ∇f[1] = 4.0 * x[1]^3 + 4.0 * x[1] * x[2] -
                44.0 * x[1] + 2.0 * x[1] + 2.0 * x[2]^2 - 14.0
            ∇f[2] = 2.0 * x[1]^2 + 2.0 * x[2] - 22.0 +
                4.0 * x[1] * x[2] + 4.0 * x[2]^3 - 28.0 * x[2]
        end

        fx = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
        if ∇f == nothing && ∇²f == nothing
            return fx
        elseif ∇²f == nothing
            return fx, ∇f
        else
            return fx, ∇f, ∇²f
        end
    end

    res = minimize!(himmelblau!, [2.0,2.0], Newton(DirectApprox()))
    res = minimize!(himmelblau!, [2.0,2.0], (Newton(DirectApprox()), NWI()))


using TRS
import NearlyNewton: NearlyExactTRSP

struct TRSolver{T} <: NearlyExactTRSP
    abstol::T
    maxiter::Integer
end
function (ms::TRSolver)(∇f::AbstractVector{T}, H, Δ, p;kwargs...) where T
    x, info = trs(Symmetric(H), ∇f, Δ)
    p .= x[:,1]
    m = dot(∇f, p) + dot(p, H * p)/2
    interior = norm(p, 2) ≤ Δ
    return (p=p, mz=m, interior=interior, λ=info.λ, hard_case=info.hard_case, solved=true)
end


h0 = zeros(2,2)
himmelblau!(h0, nothing, [2.0,2.0])

    res = minimize!(himmelblau!, [2.0,2.0], (BFGS(DirectApprox()), TRSolver(1e-8, 100)), h0)





    function polynomial!(∇²f, ∇f, x)
        if !isa(∇f, Nothing)
            ∇f[1] = -2.0 * (10.0 - x[1])
            ∇f[2] = -4.0 * (7.0 - x[2])^3
            ∇f[3] = -4.0 * (108.0 - x[3])^3
        end
        if !isa(∇²f, Nothing)
            ∇²f[1, 1] = 2.0
            ∇²f[1, 2] = 0.0
            ∇²f[1, 3] = 0.0
            ∇²f[2, 1] = 0.0
            ∇²f[2, 2] = 12.0 * (7.0 - x[2])^2
            ∇²f[2, 3] = 0.0
            ∇²f[3, 1] = 0.0
            ∇²f[3, 2] = 0.0
            ∇²f[3, 3] = 12.0 * (108.0 - x[3])^2
        end
        fx = (10.0 - x[1])^2 + (7.0 - x[2])^4 + (108.0 - x[3])^4

        if ∇f == nothing && ∇²f == nothing
            return fx
        elseif ∇²f == nothing
            return fx, ∇f
        else
            return fx, ∇f, ∇²f
        end
    end
    function polynomial!(∇f, x)
        if !isa(∇f, Nothing)
            ∇f[1] = -2.0 * (10.0 - x[1])
            ∇f[2] = -4.0 * (7.0 - x[2])^3
            ∇f[3] = -4.0 * (108.0 - x[3])^3
        end

        fx = (10.0 - x[1])^2 + (7.0 - x[2])^4 + (108.0 - x[3])^4

        if ∇f == nothing
            return fx
        else
            return fx, ∇f
        end

    end

        res = minimize!(polynomial!, [2.0,2.0,2.0], (BFGS(DirectApprox()), NWI()))




        function rosenbrock(∇²f, ∇f, x)
            if !isa(∇f, Nothing)
                ∇f[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
                ∇f[2] = 200.0 * (x[2] - x[1]^2)
            end
            if !isa(∇²f, Nothing)
                ∇²f[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
                ∇²f[1, 2] = -400.0 * x[1]
                ∇²f[2, 1] = -400.0 * x[1]
                ∇²f[2, 2] = 200.0
            end
            fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2


            if ∇f == nothing && ∇²f == nothing
                return fx
            elseif ∇²f == nothing
                return fx, ∇f
            else
                return fx, ∇f, ∇²f
            end
        end


        function rosenbrock(∇f, x)
            if !isa(∇f, Nothing)
                ∇f[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
                ∇f[2] = 200.0 * (x[2] - x[1]^2)
            end

            fx = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2


            if ∇f == nothing
                return fx
            else
                return fx, ∇f
            end
        end

        function rosenbrock_gradient!(storage::Vector, x::Vector)

        end

        function rosenbrock_hessian!(storage::Matrix, x::Vector)
            storage[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
            storage[1, 2] = -400.0 * x[1]
            storage[2, 1] = -400.0 * x[1]
            storage[2, 2] = 200.0
        end
