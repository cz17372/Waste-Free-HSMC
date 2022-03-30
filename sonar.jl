using Distributions, NPZ, Plots, StatsPlots, LinearAlgebra, ProgressMeter, JLD2, Roots
using ForwardDiff:gradient
data = npzread("sonar.npy")
function brent(f::Function, x0::Number, x1::Number, args::Tuple=();xtol::AbstractFloat=1e-7, ytol=2eps(Float64),maxiter::Integer=50)
    EPS = eps(Float64)
    y0 = f(x0,args...)
    y1 = f(x1,args...)
    if abs(y0) < abs(y1)
        # Swap lower and upper bounds.
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    end
    x2 = x0
    y2 = y0
    x3 = x2
    bisection = true
    for _ in 1:maxiter
        # x-tolerance.
        if abs(x1-x0) < xtol
            return x1
        end

        # Use inverse quadratic interpolation if f(x0)!=f(x1)!=f(x2)
        # and linear interpolation (secant method) otherwise.
        if abs(y0-y2) > ytol && abs(y1-y2) > ytol
            x = x0*y1*y2/((y0-y1)*(y0-y2)) + x1*y0*y2/((y1-y0)*(y1-y2)) + x2*y0*y1/((y2-y0)*(y2-y1))
        else
            x = x1 - y1 * (x1-x0)/(y1-y0)
        end

        # Use bisection method if satisfies the conditions.
        delta = abs(2EPS*abs(x1))
        min1 = abs(x-x1)
        min2 = abs(x1-x2)
        min3 = abs(x2-x3)
        if (x < (3x0+x1)/4 && x > x1) || (bisection && min1 >= min2/2) || (!bisection && min1 >= min3/2) || (bisection && min2 < delta) || (!bisection && min3 < delta)
            x = (x0+x1)/2
            bisection = true
        else
            bisection = false
        end

        y = f(x,args...)
        # y-tolerance.
        if abs(y) < ytol
            return x
        end
        x3 = x2
        x2 = x1
        if sign(y0) != sign(y)
            x1 = x
            y1 = y
        else
            x0 = x
            y0 = y
        end
        if abs(y0) < abs(y1)
        # Swap lower and upper bounds.
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        end
    end
    error("Max iteration exceeded")
end
y = data[:,1]; z = data[:,2:end]
logL(x) = sum(-log.(1 .+ exp.(-y.* (z*x))))
logν(x) = logpdf(Normal(0,20),x[1])+sum(logpdf.(Normal(0,5),x[2:end]))
logγ(x) = logL(x)+logν(x)
U(x) = -logγ(x); U0(x) = -logν(x); gradU(x) = gradient(U,x)
H(x,v) = U(x) + 1/2*norm(v)^2
Σ = Diagonal([[20.0^2];repeat([5.0^2],60)]);initDist = MultivariateNormal(zeros(61),Σ)
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
function ψ(x0,v0,n;ϵ,gradU)
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
    return xvec,vvec
end
function getH(x,v,H)
    n,_ = size(x)
    Hvec = zeros(n)
    for i = 1:n
        Hvec[i] = H(x[i,:],v[i,:])
    end
    return Hvec
end 
# Calculate the acceptance probability for each proposals on the trajectory
function logα(xvec,vvec,H)
    P,_ = size(xvec)
    P -= 1
    logαvec = zeros(P)
    for n = 1:P
        logαvec[n] = min(0,-H(xvec[n+1,:],vvec[n+1,:])+H(xvec[1,:],vvec[1,:]))
    end
    return logαvec
end 
function MH(x0,v0,n,ϵ,gradU,H)
    D = length(x0)
    xvec,vvec = ψ(x0,v0,n,ϵ=ϵ,gradU=gradU)
    αvec = logα(xvec,vvec,H)
    x = zeros(n,D); v = zeros(n,D)
    for i = 1:n
        if log(rand()) < αvec[i]
            x[i,:] = xvec[i+1,:]
            v[i,:] = vvec[i+1,:]
        else
            x[i,:] = xvec[1,:]
            v[i,:] = vvec[1,:]
        end
    end
    return x,v
end

function ESS(x,v,lambda,prevH,U0,U)
    N,_ = size(x)
    w = zeros(N)
    newH(x,v) = (1-lambda)*U0(x) + lambda*U(x) + 1/2*norm(v)^2
    for n = 1:N
        w[n] = -newH(x[n,:],v[n,:])+prevH(x[n,:],v[n,:])
    end
    MAX = findmax(w)[1]
    W = exp.(w.-MAX)/sum(exp.(w.-MAX))
    return 1/sum(W.^2)
end
function getW(x,v,H1,H0)
    N,_ = size(x)
    w = zeros(N)
    for n = 1:N
        w[n] = -H1(x[n,:],v[n,:])+H0(x[n,:],v[n,:])
    end
    return w
end

function SMC(N,M,U0,U,D,α,ϵ,initDist)
    # Define array to store the potential energy functions at each SMC step
    X = Array{Matrix,1}(undef,0);push!(X,zeros(N,D))
    V = Array{Matrix,1}(undef,0);push!(V,zeros(N,D))
    λ = zeros(1)
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
        A = vcat(fill.(1:N,rand(Multinomial(M,W[:,t-1])))...) # resample "starting points"
        targetU(x) = (1-λ[t-1])*U0(x) + λ[t-1]*U(x)
        targetgradU(x) = gradient(targetU,x)
        targetH(x,v) = targetU(x)+1/2*norm(v)^2
        println("Running HMC...")
        Threads.@threads for n = 1:M
            x0 = X[t-1][A[n],:]
            v0 = randn(D)
            newx,newv = MH(x0,v0,P,ϵ,targetgradU,targetH)
            for i = 1:P
                X[t][(n-1)*P+i,:] = newx[i,:]
                V[t][(n-1)*P+i,:] = newv[i,:]
            end
        end
        println("finding new λ...")
        tar(λ) = ESS(X[t],V[t],λ,targetH,U0,U) - α*N
        #newλ = find_zero(tar,(λ[end],10.0),Bisection())
        b = λ[end]+0.02
        while tar(b) >= 0.0
            println("tar(b) = ",tar(b))
            b += 0.02
        end
        println("start optimisation...")
        newλ = brent(tar,λ[end],b)
        println("the new λ is",newλ)
        if newλ >1.0
            push!(λ,1.0)
        else
            push!(λ,newλ)
        end
        newH(x,v) = (1-newλ)*U0(x) + newλ*U(x) + 1/2*norm(v)^2
        logW = hcat(logW,zeros(N))
        W = hcat(W,zeros(N))
        logW[:,t] = getW(X[t],V[t],newH,targetH)
        MAX = findmax(logW[:,t])[1]
        W[:,t] = exp.(logW[:,t] .- MAX)/sum(exp.(logW[:,t] .- MAX))  
    end
    return (X=X,λ=λ,W=W,logW=logW)
end

R = SMC(200000,200,U0,U,61,0.5,0.2,initDist)

sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))

sum(R.W[:,end].*mean(R.X[end],dims=2))