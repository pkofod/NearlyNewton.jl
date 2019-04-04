struct GradientDescent end
# might want to clean this up
update!(scheme::GradientDescent, B, A, rest...) = A
update(scheme::GradientDescent, B, A, rest...) = A
