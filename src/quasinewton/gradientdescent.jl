struct GradientDescent{T1} <: QuasiNewton{T1}
    approx::T1
end
# might want to clean this up
update!(A::UniformScaling, s, y, scheme::GradientDescent{<:InverseApprox}) = A
update!(A, s, y, scheme::GradientDescent{<:DirectApprox}) = A
update(A::UniformScaling, s, y, scheme::GradientDescent{<:InverseApprox}) = A
update(A, s, y, scheme::GradientDescent{<:DirectApprox}) = A
