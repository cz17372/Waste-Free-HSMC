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
    U0VEC[1],grad0 = U0(x0,grad=true)
    UVEC[1],grad1  = U(x0,grad=true)
    for i = 1:n
        tempv = vvec[i,:] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        xvec[i+1,:] = xvec[i,:] .+ ϵ*tempv
        U0VEC[i+1],grad0 = U0(xvec[i+1,:],grad=true)
        UVEC[i+1],grad1  = U(xvec[i+1,:],grad=true)
        vvec[i+1,:] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
    end
    return xvec,vvec,U0VEC,UVEC
end
function MH(x0,v0,n,ϵ,λ,U0,U)
    D = length(x0)
    xvec,vvec,u0vec,uvec = ψ(x0,v0,n,ϵ=ϵ,U0=U0,U=U,λ=λ)
    αvec = logα(vvec,u0vec,uvec,λ)
    x = zeros(n,D);
    u = zeros(n); u0 = zeros(n);
    for i = 1:n
        if log(rand()) < αvec[i]
            x[i,:] = xvec[i+1,:]
            u0[i] = u0vec[i+1]
            u[i]  = uvec[i+1]
        else
            x[i,:] = xvec[1,:]
            u0[i]  = u0vec[1]
            u[i]   = uvec[1]
        end
    end
    return x,u0,u
end
function ESS(U0Vec,UVec,lambda0,lambda1)
    w = (lambda1-lambda0)*U0Vec .- (lambda1-lambda0)*UVec
    MAX = findmax(w)[1]
    W = exp.(w.-MAX)/sum(exp.(w.-MAX))
    return 1/sum(W.^2)
end
function split_legs(P,nlegs)
    if rem(P,nlegs) == 0
        leg_length = repeat([div(P,nlegs)],nlegs)
    else
        leg_length = [repeat([div(P,nlegs)],nlegs-1);[P-(div(P,nlegs))*(nlegs-1)]]
    end
    return leg_length
end
function SMC(N,M,U0,U,D,α,ϵ,initDist,nlegs=1)
    X = Array{Matrix,1}(undef,0);push!(X,zeros(N,D))
    λ = zeros(1)
    U0VEC = zeros(N)
    UVEC  = zeros(N)
    logW = zeros(N,1)
    W = zeros(N,1)
    P = div(N,M)
    for i = 1:N
        X[1][i,:] = rand(initDist)
        v = randn(D)
        logW[i,1] = -U0(X[1][i,:]) - 1/2*norm(v)^2
    end
    MAX = findmax(logW[:,1])[1]
    W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
    t = 1
    while λ[end] < 1.0
        t +=1 # move to the next step
        push!(X,zeros(N,D));
        A = vcat(fill.(1:N,rand(Multinomial(M,W[:,t-1])))...)
        for n = 1:M
            x0 = X[t-1][A[n],:]
            s = 0
            leg_length=split_legs(P,nlegs)
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
        #println(tar(a),"  ",tar(b))
        newλ = find_zero(tar,(a,b),Bisection())
        if newλ >1.0
            push!(λ,1.0)
        else
            push!(λ,newλ)
        end
        #println("The new temperature is ",λ[end])
        logW = hcat(logW,zeros(N))
        W = hcat(W,zeros(N))
        logW[:,t] = (λ[t]-λ[t-1])*U0VEC .- (λ[t]-λ[t-1])*UVEC
        MAX = findmax(logW[:,t])[1]
        W[:,t] = exp.(logW[:,t] .- MAX)/sum(exp.(logW[:,t] .- MAX))
    end
    return (X=X,λ=λ,W=W,logW=logW)
end