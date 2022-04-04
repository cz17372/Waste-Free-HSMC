using Distributions, NPZ, LinearAlgebra, Roots, ProgressMeter
using ForwardDiff:gradient
data = npzread("sonar.npy");y = data[:,1]; z = data[:,2:end]
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
U0(x) = -logν(x); gradU0(x) = gradient(U0,x);
Σ = Diagonal([[20.0^2];repeat([5.0^2],60)]);
initDist = MultivariateNormal(zeros(61),Σ)
function HMC(x0,N;L,ϵ,H,gradU)
    D = length(x0)
    X = zeros(N+1,D)
    X[1,:] = x0
    acc = 0
    for n = 1:N
        v0 = rand(Normal(0,1),D)
        propx,propv = leapfrog(X[n,:],v0,n=L,ϵ=ϵ,gradU=gradU)
        alpha = min(0,-H(X[n,:],v0)+H(propx,propv))
        if log(rand()) < alpha
            X[n+1,:] = propx
            acc += 1
        else
            X[n+1,:] = X[n,:]
        end
        if rem(n,100) == 0
            println("average acceptance prob = ",acc/n)
        end
    end
    return X
end
function leapfrog(x0,v0;n,ϵ,gradU)
    D = length(x0)
    xvec = zeros(n+1,D)
    vvec = zeros(n+1,D)
    xvec[1,:] = x0
    vvec[1,:] = v0
    for i = 1:n
        tempv = vvec[i,:] .- ϵ/2*gradU(xvec[i,:])
        xvec[i+1,:] = xvec[i,:] .+ ϵ*tempv
        vvec[i+1,:] = tempv .- ϵ/2*gradU(xvec[i+1,:])
    end
    return xvec[end,:],vvec[end,:]
end
function split_legs(P,nlegs)
    if rem(P,nlegs) == 0
        leg_length = repeat([div(P,nlegs)],nlegs)
    else
        leg_length = [repeat([div(P,nlegs)],nlegs-1);[P-(div(P,nlegs))*(nlegs-1)]]
    end
    return leg_length
end
include("sonar.jl")
N = 10000
M = 100
D = 61
α = 0.5
ϵ = 0.2
nlegs = 10
P = div(N,M)
X = Array{Matrix,1}(undef,0);push!(X,zeros(N,D))
λ = zeros(1)
U0VEC = zeros(N)
UVEC  = zeros(N)
logW = zeros(N,1)
W = zeros(N,1)
for i = 1:N
    X[1][i,:] = rand(initDist)
    v = randn(D)
    logW[i,1] = -U0(X[1][i,:]) 
end
MAX = findmax(logW[:,1])[1]
W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
t = 1
t +=1 # move to the next step
push!(X,zeros(N,D));
A = vcat(fill.(1:N,rand(Multinomial(M,W[:,t-1])))...)

@time for n = 1:M
    println(n)
    s = 0
    leg_length=split_legs(P,nlegs)
    x0 = X[t-1][A[n],:]
    for j = 1:nlegs
        v0 = randn(D)
        X[t][((n-1)*P+s+1):((n-1)*P+s+leg_length[j]),:],U0VEC[((n-1)*P+s+1):((n-1)*P+s+leg_length[j])],UVEC[((n-1)*P+s+1):((n-1)*P+s+leg_length[j])] = MH(x0,v0,leg_length[j],ϵ,λ[end],U0,U)
        s += leg_length[j]
    end
end
tar(l) = ESS(U0VEC,UVEC,λ[end],l) - α*N
a = λ[end]
b = λ[end]+0.1
       
while tar(a)*tar(b) >= 0
    b += 0.1
end
newλ = find_zero(tar,(a,b),Bisection())
if newλ >1.0
    push!(λ,1.0)
else
    push!(λ,newλ)
end
logW = hcat(logW,zeros(N))
W = hcat(W,zeros(N))
logW[:,t] = (λ[t]-λ[t-1])*U0VEC .- (λ[t]-λ[t-1])*UVEC
MAX = findmax(logW[:,t])[1]
W[:,t] = exp.(logW[:,t] .- MAX)/sum(exp.(logW[:,t] .- MAX))

