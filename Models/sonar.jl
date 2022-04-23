module sonar
using Distributions,NPZ,LinearAlgebra
using ForwardDiff:gradient
D = 61
dir = @__DIR__
data = npzread(dir*"/sonar.npy"); y = data[:,1]; z = data[:,2:end]
function logL(x;grad=false)
    elvec = exp.(-y .* (z*x))
    llk = sum(-log.(1 .+ elvec))
    if grad
        g = -sum((-1 .+ 1 ./ (1 .+ elvec)) .* (y.*z),dims=1)[1,:]
        return (llk,g)
    else
        return llk
    end
end
logν(x) = logpdf(Normal(0,20),x[1])+sum(logpdf.(Normal(0,5),x[2:end]))
logγ(x) = logL(x)+logν(x)
function U(x;grad=false)
    if grad
        llk,g1 = logL(x,grad=grad)
        u = -logν(x)-llk
        g2 = -gradient(logν,x) .- g1
        return (u,g2)
    else
        llk = logL(x,grad=grad)
        return -logν(x) - llk
    end
end
function U0(x;grad=false)
    if grad
        llk = -logν(x)
        g = -gradient(logν,x)
        return (llk,g)
    else
        return -logν(x)
    end
end
Σ = Diagonal([[20.0^2];repeat([5.0^2],60)]);
initDist = MultivariateNormal(zeros(61),Σ)
export U0, U, initDist
end