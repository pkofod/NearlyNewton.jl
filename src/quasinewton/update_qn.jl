import Base.circshift!

struct QNCache{T1, T2}
    y::T1 # change in successive gradients
    d::T2 # search direction
    s::T2 # change in x
end
function QNCache(x, g)
    QNCache(copy(g), copy(x), copy(x))
end
function preallocate_qn_caches_inplace(x0)
    # Maintain gradient and state pairs in QNCache
    cache = QNCache(x0, x0)
    return cache
end


function update_qn!(cache::QNCache, objective, ∇fx, z, ∇fz, B, scheme, scale=nothing)
    d, s, y= cache.d, cache.s, cache.y
    fz, ∇fz = objective(∇fz, z)
    # add Project gradient

    # Update y
    @. y = ∇fz - ∇fx

    # Update B
    if scale == nothing
        if isa(scheme.approx, DirectApprox)
            Badj = dot(y, d)/dot(y, y).*B
        else
            Badj = dot(y, y)/dot(y, d).*B
        end
    else
        Badj = B
    end

    # Quasi-Newton update
    B = update!(Badj, s, y, scheme)

    return fz, ∇fz, B
end

function update_qn!(cache::QNCache, objective, ∇fx, z, ∇fz, B, scheme::Newton, scale=nothing)
    fz, ∇fz, B = objective(B, ∇fz, z)

    return fz, ∇fz, B
end

function update_qn(objective, d, s, ∇fx, z, ∇fz, B, scheme, scale=nothing)
    fz, ∇fz = objective(∇fz, z)
    # add Project gradient

    # Update y
    y = ∇fz - ∇fx

    # Update B
    if scale == nothing
        if isa(scheme.approx, DirectApprox)
            Badj = dot(y, d)/dot(y, B* y)*B
        else
            Badj = dot(y, B* y)/dot(y, d)*B
        end
    else
         Badj = B
    end

    # Quasi-Newton update
    B = update(Badj, s, y, scheme)

    return fz, ∇fz, B
end

function update_qn(objective, d, s, ∇fx, z, ∇fz, B, scheme::Newton, is_first=nothing)
    fz, ∇fz, B = objective(B, ∇fx, z)

    return fz, ∇fz, B
end
