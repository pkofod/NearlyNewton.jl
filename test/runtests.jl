using NearlyNewton
using Test
using StaticArrays

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
    quasinewton(f, g, x0, BFGS(); inverse = true)
    quasinewton(f, g, x0, BFGS(); inverse = false)

    quasinewton(f, g, x0, DFP(); inverse = true)
    quasinewton(f, g, x0, DFP(); inverse = false)

    quasinewton(f, g, x0, SR1(); inverse = true)
    quasinewton(f, g, x0, SR1(); inverse = false)

    quasinewton(f, g, rand(3), GradientDescent(); inverse = true)
    quasinewton(f, g, rand(3), GradientDescent(); inverse = false)


    x0s = @SVector [0.1, 0.1, 0.1]
    quasinewton(f, gs, x0s, GradientDescent(); inverse = true)
    quasinewton(f, gs, x0s, GradientDescent(); inverse = false)

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
    quasinewton(himmelblau, himmelblau_gradient, x0, BFGS(); inverse = true)
    quasinewton(himmelblau, himmelblau_gradient, x0, BFGS(); inverse = false)

    quasinewton(himmelblau, himmelblau_gradient, x0, DFP(); inverse = true)
    quasinewton(himmelblau, himmelblau_gradient, x0, DFP(); inverse = false)

    quasinewton(himmelblau, himmelblau_gradient, x0, SR1(); inverse = true)
    quasinewton(himmelblau, himmelblau_gradient, x0, SR1(); inverse = false)

    quasinewton(himmelblau, himmelblau_gradient, rand(2), GradientDescent(); inverse = true)
    quasinewton(himmelblau, himmelblau_gradient, rand(2), GradientDescent(); inverse = false)

    x0s = @SVector [3.0, 1.0]
    quasinewton(himmelblau, himmelblau_gradients, x0s, GradientDescent(); inverse = true)
    quasinewton(himmelblau, himmelblau_gradients, x0s, GradientDescent(); inverse = false)

    x0 = [3.0, 1.0]
    quasinewton!(himmelblau, himmelblau_gradient!, x0, BFGS(); inverse = true)
    quasinewton!(himmelblau, himmelblau_gradient!, x0, BFGS(); inverse = false)

    quasinewton!(himmelblau, himmelblau_gradient!, x0, DFP(); inverse = true)
    quasinewton!(himmelblau, himmelblau_gradient!, x0, DFP(); inverse = false)

    quasinewton!(himmelblau, himmelblau_gradient!, x0, SR1(); inverse = true)
    quasinewton!(himmelblau, himmelblau_gradient!, x0, SR1(); inverse = false)

    quasinewton!(himmelblau, himmelblau_gradient!, rand(2), GradientDescent(); inverse = true)
    quasinewton!(himmelblau, himmelblau_gradient!, rand(2), GradientDescent(); inverse = false)

    x0s = @SVector [3.0, 1.0]
    quasinewton(himmelblau, himmelblau_gradients, x0s, GradientDescent(); inverse = true)
    quasinewton(himmelblau, himmelblau_gradients, x0s, GradientDescent(); inverse = false)


end
