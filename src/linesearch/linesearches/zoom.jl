# algorithm 3.6 in N&W
function zoom(ϕ, ϕ0, α0)
    function interpolate(a_i1::Real, a_i::Real,
                     ϕ_a_i1::Real, ϕ_a_i::Real,
                     dϕ_a_i1::Real, dϕ_a_i::Real)
        d1 = dϕ_a_i1 + dϕ_a_i - 3 * (ϕ_a_i1 - ϕ_a_i) / (a_i1 - a_i)
        d2 = sqrt(d1 * d1 - dϕ_a_i1 * dϕ_a_i)
        return a_i - (a_i - a_i1) *((dϕ_a_i + d2 - d1) /(dϕ_a_i - dϕ_a_i1 + 2 * d2))
    end
    ϕj = ϕ0
    αj = α0
    for i = 1:10
        interpolate()
        if

        else

        end

    end
    return αj
end
