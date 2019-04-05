"""
First update is phi = 0, so BFGS for hessian
"""
function first_update(X, p, q)
    X + q*q'/dot(p, q) - X*q*q'*X/(q'*X*q)
end
function second_update(X, p, q)
    (I - p*q'/dot(q, p))*X*(I - q*p'/dot(q, p)) + p*p'/dot(q, p)
end
function first_update!(A, p, q)
    A .= A + q*q'/dot(p, q) - A*q*q'*A/(q'*A*q)
end
function second_update!(A, p, q)
    A .= (I - p*q'/dot(q, p))*A*(I - q*p'/dot(q, p)) + p*p'/dot(q, p)
end
