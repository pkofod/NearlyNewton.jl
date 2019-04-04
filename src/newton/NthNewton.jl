mutable struct NthNewton <: NearlyNewtonMethod
   counter::Int
   Nth::Int
end
NthNewton(Nth) = NthNewton(Nth-1, Nth)
function update!(scheme::NthNewton, B::DirectApprox, ∇²f, x)
   scheme.counter += 1
   if scheme.count % scheme.Nth == 0
      B.A = ∇²f
      scheme.counter = 0
   end
   B.A
end
