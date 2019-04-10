"""
First update is phi = 0, so BFGS for hessian
"""
function first_update(X, p, q)
    X + q*q'/dot(p, q) - X*q*q'*X/(q'*X*q)
end
function second_update(X, p, q)
    dqp = dot(q, p)
    (I - p*q'/dqp)*X*(I - q*p'/dqp) + p*p'/dqp
end
function first_update!(A, p, q)
    qqt = q*q'
    A .+= qqt/dot(p, q) - A*qqt*A/(q'*A*q)
end
function second_update!(A, p, q)
    dqp = dot(q, p)
    A .= (I - p*q'/dqp)*A*(I - q*p'/dqp) + p*p'/dqp
end
