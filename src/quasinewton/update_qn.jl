import Base.circshift!

struct QNCache{T1, T2}
    ∇f_curr::T1 # gradient before step is taken
    ∇f_next::T1 # gradient after step is taken
    y::T1 # change in successive gradients
    x_curr::T2 # x before step is taken
    x_next::T2 # x after step is taken
    d::T2 # search direction
    s::T2 # final step
end
function QNCache(x, g)
    QNCache(copy(g), copy(g), copy(g), copy(x), copy(x), copy(x), copy(x))
end
function preallocate_qn_caches_inplace(x0)
    # Maintain gradient and state pairs in QNCache
    cache = QNCache(x0, x0)
    return cache
end


function shift!(qnc::QNCache)
    copyto!(qnc.∇f_curr, qnc.∇f_next)
    copyto!(qnc.x_curr, qnc.x_next)
end


function update_qn!(cache::QNCache, B, scheme, is_first=nothing)
    d, s, y, ∇f_next, ∇f_curr = cache.d, cache.s, cache.y, cache.∇f_next, cache.∇f_curr
    # Update y
    @. y = ∇f_next - ∇f_curr

    # Update B
    if isa(is_first, Nothing)
        Badj = dot(y, d)/dot(y, y)*I
    else
        Badj = B
    end
    # Quasi-Newton update
    B = update!(Badj, s, y, scheme)

    return B
end

function update_qn(d, s, ∇f_curr, ∇f_next, B, scheme, is_first=nothing)
    # Update y
    y = @. ∇f_next - ∇f_curr

    # Update B
    if isa(is_first, Nothing)
        Badj = dot(y, d)/dot(y, y)*B
    else
        Badj = B
    end

    # Quasi-Newton update
    B = update(Badj, s, y, scheme)
end
