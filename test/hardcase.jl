# verify that solve_tr_subproblem! finds the minimum
using NearlyNewton, Test
using LinearAlgebra: norm, dot


trsolver = NWI()

n = 2
gr = [-0.74637,0.52388]
# H is not posdef!
H = [0.945787 -3.07884; -3.07884 -1.27762]


s = zeros(n)
Δ = 1.0
solution = trsolver(gr, H, Δ, s ;maxiter=100)
println("Trust region : $Δ")
println("Norm of step : $(norm(solution.p,2))")

for j in 1:100000
    bad_s = rand(n)
    bad_s ./= norm(bad_s)  # boundary
    model(s2) = dot(gr, s2) + (dot(s2, H * s2))/2
    @test model(s) <= model(bad_s) + 1e-8
end

# H is now posdef!
Hp = H*H
s = zeros(n)
Δ = 1.0
solution = trsolver(gr, Hp, Δ, s ;maxiter=100)
println("Trust region : $Δ")
println("Norm of step : $(norm(solution.p,2))")
println("Is it interior? $(solution.interior)")


for j in 1:100000
    bad_s = rand(n)
    if norm(bad_s) > Δ
        bad_s ./= norm(bad_s)  # boundary
    end
    model(s2) = dot(gr, s2) + (dot(s2, Hp * s2))/2
    @test model(s) <= model(bad_s) + 1e-8
end

Hhard = [-0.945787 0.0; 0.0 -1.27762]
Δ = 1.0
solution = trsolver(gr, Hhard, Δ, s ;maxiter=100)


or

# From tR book p180
s2 = zeros(4)
gr2 = [0.0, 1.0, 1.0, 1.0]
Hhard2 = [-2.0 0.0 0.0 0.0;0.0 -1.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 1.0]
solution = trsolver(gr2,  Hhard2, Δ, s2 ;maxiter=100)
