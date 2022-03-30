using Distributions, Plots, StatsPlots, LinearAlgebra
using ForwardDiff:gradient
U(x) = -logpdf(Normal(0,2),x[1]) - logpdf(Normal(sin(x[1]),sqrt(1/100)),x[2])
gradU(x) = gradient(U,x)
# Define the Hanmiltonian H(x,v) = q(x) + p(v) 
H(x,v) = U(x) + 1/2*norm(v)^2
# define the deterministic transformation used for Hamiltonian proposals
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
function HMC(x0,N;L,ϵ,H,gradU)
    D = length(x0)
    X = zeros(N+1,D)
    X[1,:] = x0
    for n = 1:N
        v0 = rand(Normal(0,1),D)
        propx,propv = leapfrog(X[n,:],v0,n=L,ϵ=ϵ,gradU=gradU)
        alpha = min(0,-H(X[n,:],v0)+H(propx,propv))
        if log(rand()) < alpha
            X[n+1,:] = propx
        else
            X[n+1,:] = X[n,:]
        end
    end
    return X
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
Normalize(x) = x / sum(x)
function WFSMC(M,P,tempvec,D;ϵ,U0,U,initDist)
    # Define the sequence of distributions 
    T = length(tempvec)
    HVec = Array{Any,1}(undef,T)
    gradUVec = Array{Any,1}(undef,T)
    for i = 1:T
        temperedU(x) = (1.0 - tempvec[i])*U0(x) + tempvec[i] * U(x)
        gradUVec[i] = x -> gradient(temperedU,x)
        HVec[i] = (x,v) -> temperedU(x) + 1/2*norm(v)^2
    end
    N = M * P
    X = zeros(N,D,T)
    V = zeros(N,D,T)
    W = zeros(N,T)
    logW = zeros(N,T)
    for i = 1:N
        X[i,:,1] = rand(initDist)
        V[i,:,1] = rand(Normal(0,1),D)
        logW[i,1] = -HVec[1](X[i,:,1],V[i,:,1])
    end
    MAX = findmax(logW[:,1])[1]
    W[:,1] = Normalize(exp.(logW[:,1] .- MAX))
    for t = 2:T
        A = vcat(fill.(1:N,rand(Multinomial(M,W[:,t-1])))...)
        for n = 1:M
            x0 = X[A[n],:,t-1]
            v0 = rand(Normal(0,1),D)
            newx,newv  = MH(x0,v0,P,ϵ,gradUVec[t-1],HVec[t-1])
            for i = 1:P
                X[(n-1)*P+i,:,t] = newx[i,:]
                V[(n-1)*P+i,:,t] = newv[i,:]
                logW[(n-1)*P+i,t] = -HVec[t](newx[i,:],newv[i,:])+HVec[t-1](newx[i,:],newv[i,:])
            end
        end
        MAX = findmax(logW[:,t])[1]
        W[:,t] = Normalize(exp.(logW[:,t] .- MAX))
    end
    return (X=X,W=W,V=V)
end
U0(x) = -sum(logpdf.(Normal(0,2),x))
initDist = MultivariateNormal([0.0,0.0],4.0*I)
R = WFSMC(200,50,collect(0:0.1:1.0),2,ϵ=0.05,U0=U0,U=U,initDist=initDist)
A = vcat(fill.(1:10000,rand(Multinomial(10000,R.W[:,end])))...)
X = R.X[A,:,end]
scatter(X[:,1],X[:,2],markersize=2.0,markerstrokewidth=0.0)
density(X[:,2])