module banana
using Distributions, LinearAlgebra


D = 2
function U(x;requires_grad=true)
    llk = 0.5*(0.03*x[1]^2+(x[2]+0.03*(x[1]^2-100))^2)
    if requires_grad
        grad = [0.5*(0.06*x[1]+0.12*x[1]*(x[2]+0.03*(x[1]^2-100))),x[2]+0.03*(x[1]^2-100)]
        return llk,grad
    else
        return llk
    end
end

end