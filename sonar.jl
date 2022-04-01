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
function logα(vvec,U0VEC,UVEC,λ)
    P,_ = size(vvec)
    P -= 1
    logαvec = zeros(P)
    for n = 1:P
        logαvec[n] = min(0,-(1-λ)*U0VEC[n+1]-λ*UVEC[n+1] - 1/2*norm(vvec[n+1,:])^2 + (1-λ)*U0VEC[1] + λ*UVEC[1] + 1/2*norm(vvec[1,:])^2)
    end
    return logαvec
end
function ψ(x0,v0,n;ϵ,U0,U,λ)
    D = length(x0)
    xvec = zeros(n+1,D)
    vvec = zeros(n+1,D)
    xvec[1,:] = x0
    vvec[1,:] = v0
    U0VEC = zeros(n+1)
    UVEC  = zeros(n+1)
    U0VEC[1] = U0(x0);grad0 = gradient(U0,x0)
    UVEC[1],grad1  = U(x0,grad=true)
    for i = 1:n
        tempv = vvec[i,:] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        xvec[i+1,:] = xvec[i,:] .+ ϵ*tempv
        U0VEC[i+1] = U0(xvec[i+1,:]);grad0 = gradient(U0,xvec[i+1,:])
        UVEC[i+1],grad1  = U(xvec[i+1,:],grad=true)
        vvec[i+1,:] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
    end
    return xvec,vvec,U0VEC,UVEC
end
function MH(x0,v0,n,ϵ,λ,U0,U)
    grad(x) = (1-λ)*gradU0(x) + λ*gradU(x)
    D = length(x0)
    xvec,vvec,u0vec,uvec = ψ(x0,v0,n,ϵ=ϵ,U0=U0,U=U,λ=λ)
    αvec = logα(vvec,u0vec,uvec,λ)
    x = zeros(n,D); v = zeros(n,D)
    u = zeros(n); u0 = zeros(n);
    for i = 1:n
        if log(rand()) < αvec[i]
            x[i,:] = xvec[i+1,:]
            v[i,:] = vvec[i+1,:]
            u0[i] = u0vec[i+1]
            u[i]  = uvec[i+1]
        else
            x[i,:] = xvec[1,:]
            v[i,:] = vvec[1,:]
            u0[i]  = u0vec[1]
            u[i]   = uvec[1]
        end
    end
    return x,v,u0,u
end
function ESS(U0Vec,UVec,lambda0,lambda1)
    w = (lambda1-lambda0)*U0Vec .- (lambda1-lambda0)*UVec
    MAX = findmax(w)[1]
    W = exp.(w.-MAX)/sum(exp.(w.-MAX))
    return 1/sum(W.^2)
end
function SMC(N,M,U0,U,D,α,ϵ,initDist)
    X = Array{Matrix,1}(undef,0);push!(X,zeros(N,D))
    V = Array{Matrix,1}(undef,0);push!(V,zeros(N,D))
    λ = zeros(1)
    U0VEC = zeros(N)
    UVEC  = zeros(N)
    logW = zeros(N,1)
    W = zeros(N,1)
    P = div(N,M)
    for i = 1:N
        X[1][i,:] = rand(initDist)
        V[1][i,:] = randn(D)
        logW[i,1] = -U0(X[1][i,:]) - 1/2*norm(V[1][i,:])^2
    end
    MAX = findmax(logW[:,1])[1]
    W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
    t = 1
    while λ[end] < 1.0
        t +=1 # move to the next step
        push!(X,zeros(N,D));push!(V,zeros(N,D))
        A = vcat(fill.(1:N,rand(Multinomial(M,W[:,t-1])))...)
        for n = 1:M
            x0 = X[t-1][A[n],:]
            v0 = randn(D)
            newx,newv,u0vec,uvec = MH(x0,v0,P,ϵ,λ[end],U0,U)
            X[t][((n-1)*P+1):(n*P),:] = newx
            V[t][((n-1)*P+1):(n*P),:] = newv
            U0VEC[((n-1)*P+1):(n*P)]  = u0vec
            UVEC[((n-1)*P+1):(n*P)]   = uvec
        end
        tar(l) = ESS(U0VEC,UVEC,λ[end],l) - α*N
        newλ = find_zero(tar,(λ[end],10.0),Bisection())
        if newλ >1.0
            push!(λ,1.0)
        else
            push!(λ,newλ)
        end
        println("The new temperature is ",λ[end])
        logW = hcat(logW,zeros(N))
        W = hcat(W,zeros(N))
        logW[:,t] = (λ[t]-λ[t-1])*U0VEC .- (λ[t]-λ[t-1])*UVEC
        MAX = findmax(logW[:,t])[1]
        W[:,t] = exp.(logW[:,t] .- MAX)/sum(exp.(logW[:,t] .- MAX))
    end
    return (X=X,λ=λ,W=W,logW=logW)
end