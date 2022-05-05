using Distributions,LinearAlgebra
model = include("Models/sonar.jl")
getKE(v,invM) = 1/2*transpose(v)*invM*v
getPE(U0,U,λ) = (1-λ)*U0 + λ*U
norm_vec(w)   = w/sum(w)
mutable struct LeapFrogTrajectory
    X::Matrix{Float64}
    V::Matrix{Float64}
    U0::Vector{Float64}
    U::Vector{Float64}
end
mutable struct Hamiltonian
    PE::Vector{Float64}
    KE::Vector{Float64}
end
function getHamiltonian(LF::LeapFrogTrajectory,λ::Float64,invM::Matrix{Float64})
    T = length(LF.U)
    PE = zeros(T)
    KE = zeros(T)
    for t = 1:T
        PE[t] = getPE(LF.U0[t],LF.U[t],λ)
        KE[t] = 1/2*transpose(LF.V[:,t])*invM*LF.V[:,t]
    end
    return Hamiltonian(PE,KE)
end
function ψ(x0::Vector{Float64},v0::Vector{Float64},niter::Int64,ϵ::Float64,λ::Float64,invM::Matrix{Float64},model)
    X = zeros(model.D,niter+1)
    V = zeros(model.D,niter+1)
    X[:,1] = x0
    V[:,1] = v0
    U0 = zeros(niter+1)
    U  = zeros(niter+1)
    U0[1],grad0 = model.U0(x0,requires_grad=true)
    U[1],grad1  = model.U(x0,requires_grad=true)
    for i = 1:niter
        tempv = V[:,i] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        X[:,i+1] = X[:,i] .+ ϵ*invM*tempv
        U0[i+1],grad0 = model.U0(X[:,i+1],requires_grad=true)
        U[i+1],grad1  = model.U(X[:,i+1],requires_grad=true)
        V[:,i+1] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
    end
    return LeapFrogTrajectory(X,V,U0,U)
end
function LogTrajectoryWeights(LF::LeapFrogTrajectory,λ0::Float64,λ1::Float64,invM::Float64)
    initH = (1-λ0)*LF.U0[1] + λ0*LF.U[1] + getKE(LF.V[1,:],invM)
    newH  = getHamiltonian(LF,λ1,invM)
    return -newH.PE .- newH.KE .+ initH
end
function getESS(logW::Vector{Vector{Float64}})
    res = zeros(0)
    for vec in logW
        res = [res;vec]
    end
    res = norm_vec(exp.(res .- findmax(res)[1]))
    return 1/sum(res.^2)
end
M = 100; ϵ = 0.1; P = 200; N = M*P
X = Matrix{LeapFrogTrajectory}(undef,M,0)
λ = zeros(1)
logW = zeros(N,1)
W = zeros(N,1)
X0 = rand(model.initDist,M)
t = 1
if t == 1
    start_points =X0
end
X = hcat(X,Vector{LeapFrogTrajectory}(undef,M))
Mat = diagm(ones(model.D))
@time Threads.@threads for m = 1:M
    v0 = rand(MultivariateNormal(zeros(model.D),Mat))
    X[m,t] = ψ(start_points[:,m],v0,P-1,ϵ,λ[t],Mat,model)
end