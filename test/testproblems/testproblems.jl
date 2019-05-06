import NearlyNewton: minimize, minimize!
struct TestProblem
    objective
    initial
    B0
    opts
end
minimize!(tp::TestProblem, method; x0=tp.initial, B0=tp.B0, opts=tp.opts) = minimize!(tp.objective, x0, method, B0, opts)
minimize(tp::TestProblem, method; x0=tp.initial, B0=tp.B0, opts=tp.opts) = minimize(tp.objective, x0, method, B0, opts)
