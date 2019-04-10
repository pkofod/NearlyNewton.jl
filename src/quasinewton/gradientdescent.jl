struct GradientDescent end
# might want to clean this up
update!(scheme::GradientDescent, B::InverseApprox, A::UniformScaling, rest...) = A
update!(scheme::GradientDescent, B::DirectApprox, A, rest...) = A
update(scheme::GradientDescent, B::InverseApprox, A::UniformScaling, rest...) = A
update(scheme::GradientDescent, B::DirectApprox, A, rest...) = A
