module sonar
using Distributions,NPZ,LinearAlgebra
using ForwardDiff:gradient
D = 61
dir = @__DIR__
data = npzread(dir*"/sonar.npy"); y = data[:,1]; z = data[:,2:end]
function logL(x;requires_grad=true)
    elvec = exp.(-y .* (z*x))
    llk = sum(-log.(1 .+ elvec))
    if requires_grad
        grad = -sum((-1 .+ 1 ./ (1 .+ elvec)) .* (y.*z),dims=1)[1,:]
        return (llk,grad)
    else
        return llk
    end
end
function logν(x;requires_grad=true)
    σ = [[20.0^2];repeat([5.0^2],60)]
    if requires_grad
        log_density = -1/2*sum((x.^2) ./ σ) - 1/2*sum(log.(2*pi*σ))
        grad = -x ./ σ
        return (log_density,grad)
    else
        return -1/2*sum((x.^2) ./ σ) - 1/2*sum(log.(2*pi*σ))
    end
end
function U(x;requires_grad=true)
    if requires_grad
        llk,grad_llk = logL(x,requires_grad=true)
        prior,grad_prior = logν(x,requires_grad=true)
        u = -llk - prior
        grad = -grad_llk .- grad_prior
        return (u,grad)
    else
        llk = logL(x,requires_grad=false)
        prior = logν(x,requires_grad=false)
        return -llk - prior
    end
end
function U0(x;requires_grad=true)
    if requires_grad
        u0,grad = logν(x,requires_grad=true)
        return (-u0,-grad)
    else
        u0 = logν(x,requires_grad=false)
        return -u0
    end
end
Σ = Diagonal([[20.0^2];repeat([5.0^2],60)]);
initDist = MultivariateNormal(zeros(D),Σ)
export U0, U, initDist
end