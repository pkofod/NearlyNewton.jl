"""
First update is phi = 0, so BFGS for hessian
"""
function first_update(X, p, q)
    X + q*q'/dot(p, q) - X*q*q'*X/(q'*X*q)
end
function second_update(X, p, q)
    (I - p*q'/dot(q, p))*X*(I - q*p'/dot(q, p)) + p*p'/dot(q, p)
end
