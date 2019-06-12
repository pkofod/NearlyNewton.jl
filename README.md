# Optim.jl

What are the rules of NearlyNewton?

* Don't be a mess!
* Support reasonable number types!
* Support all arrays!
* In-place and out-of-place for all situations!
* Fail softly to facilitate diagnosis!
* Be generic (linsolves, linear operators, ...)
* Be fast!

# Examples

## Scalar optimization (w/ different number types)


## Multivariate optimization (w/ different number and array types)


## Custom solve

# TODO
More line searches
Provide better output? See what is necessary
Tracing
Max step in line search
Curve fit max https://github.com/JuliaNLSolvers/Optim.jl/issues/207
Initial hessian
Univariate bounds
AD??
Conjugate gradients
tol for nelder mead?
Mention closure in docs
Check that we have a direction of decent before entereing line search
time limits?
AdaDelta? AdaGrad? Rprop? http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332
Momentum methods?
Reset directions?
https://github.com/JuliaNLSolvers/Optim.jl/issues/153
tuples of things ? https://github.com/JuliaNLSolvers/Optim.jl/issues/160
fminbox value https://github.com/JuliaNLSolvers/Optim.jl/issues/167 https://github.com/JuliaNLSolvers/Optim.jl/issues/541

https://github.com/multidis/SATsallis.jl
preconditioning! https://github.com/JuliaNLSolvers/Optim.jl/issues/188 https://github.com/JuliaNLSolvers/Optim.jl/issues/202
limits to f, g, h calls
parallel nelder mead https://github.com/JuliaNLSolvers/Optim.jl/issues/268
Subplex
Check numbers! Bigfloat etc
maximize vs minimize
initial step size
Hessian vector product
Stop at some bound
Manifold optimization! https://github.com/JuliaNLSolvers/Optim.jl/issues/448
Complex optimization! https://github.com/JuliaNLSolvers/Optim.jl/issues/438 https://github.com/JuliaNLSolvers/Optim.jl/pull/440 https://github.com/JuliaNLSolvers/Optim.jl/issues/470
right preconditioning https://github.com/JuliaNLSolvers/Optim.jl/issues/477
AGD https://github.com/JuliaNLSolvers/Optim.jl/issues/458
Arbitratry precision
scalar bounds in fminbox
Nested optimization?
Allow factorizations god dmamit
Allow dual numbers? https://github.com/JuliaNLSolvers/Optim.jl/issues/248
Nonlinear equations!
Recursive arrays
Progressmeter https://github.com/JuliaNLSolvers/Optim.jl/issues/442
using ProgressMeter
prog = ProgressThresh(1e-8, "Minimizing:")
function f()
df(x) = 4*x^3 - 2*x + 0.8
x = 1.0
h = 0.1
while df(x) > 1e-8
  x -= h * df(x)
  ProgressMeter.update!(prog, abs(df(x)))
  sleep(0.3)
end
end
f()
behavior or manifolds and complex https://github.com/JuliaNLSolvers/Optim.jl/issues/512

NGMRES https://github.com/JuliaNLSolvers/Optim.jl/issues/514 https://github.com/JuliaNLSolvers/Optim.jl/issues/515 https://github.com/JuliaNLSolvers/Optim.jl/issues/573

L-BFGS-B https://github.com/JuliaNLSolvers/Optim.jl/issues/521
Box manifodl https://github.com/JuliaNLSolvers/Optim.jl/issues/532
LIPO https://github.com/JuliaNLSolvers/Optim.jl/issues/539
TR subfunction should handle hard case https://github.com/JuliaNLSolvers/Optim.jl/issues/540

rand_search = https://github.com/JuliaNLSolvers/Optim.jl/issues/550
SAMIN https://github.com/JuliaNLSolvers/Optim.jl/issues/555

CMAES https://github.com/JuliaNLSolvers/Optim.jl/issues/574

error if integers https://github.com/JuliaNLSolvers/Optim.jl/issues/576
linesearch warnings/error https://github.com/JuliaNLSolvers/Optim.jl/issues/579

iterator https://github.com/JuliaNLSolvers/Optim.jl/issues/599
fminbox 2order https://github.com/JuliaNLSolvers/Optim.jl/issues/600  
ipnewton improvements https://github.com/JuliaNLSolvers/Optim.jl/issues/609 https://github.com/JuliaNLSolvers/Optim.jl/issues/651
should we have a lower bound on the step size ? https://github.com/JuliaNLSolvers/Optim.jl/issues/631

project gradient in reset https://github.com/JuliaNLSolvers/Optim.jl/issues/649
optimize/minimize/maximize https://github.com/JuliaNLSolvers/Optim.jl/issues/685
could do maximize as
```
struct MaxWrap
f
end
function (mw::MaxWrap)(h, g, x)
    out = mx.f(x)
    flip sign
    ...
end
```
unitful -> wait for linalg

caches should be possible for in ething to od https://github.com/JuliaNLSolvers/Optim.jl/issues/704
