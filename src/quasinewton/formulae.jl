"""
First update is phi = 0, so BFGS for hessian
"""
function first_update(X, p, q)
    X + p*p'/dot(p, q) - X*q*q'*X/(q'*X*q)
end
function second_update(X, p, q)
    dqp = dot(q, p)
    (I - p*q'/dqp)*X*(I - q*p'/dqp) + p*p'/dqp
end
function first_update!(A, p, q)
    A .+= p*p'/dot(p, q) - A*(q*q')*A/(q'*A*q)
end
function second_update!(A, p, q)
    dqp = dot(q, p)
    # so right now, we just skip the update if dqp is zero
    # but we might do something else here
    if dqp > 0.0
        A .= (I - p*q'/dqp)*A*(I - q*p'/dqp) + p*p'/dqp
    end
    A
end
