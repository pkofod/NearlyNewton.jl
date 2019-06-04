include("linesearch/ls_optimize.jl")

using InteractiveUtils

function trace_show(show_trace, f_curr, f_next, x_next, x_curr, α)
    if show_trace
        println("Objective value (curr): ", f_next)
        println("Objective value (prev): ", f_curr)
        println("Step size: ", α)
    end
end
