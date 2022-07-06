module wasterecycling
using Distributions, Roots, LinearAlgebra, StatsBase, Optim
function ψ(x0,v0,n,invM;ϵ,model,λ)
    xvec = zeros(model.D,n)
    vvec = zeros(model.D,n)
    xvec[:,1] = x0
    vvec[:,1] = v0
    U0VEC = zeros(n)
    UVEC  = zeros(n)
    U0VEC[1],grad0 = model.U0(x0,requires_grad=true)
    UVEC[1],grad1  = model.U(x0,requires_grad=true)
    KEVEC = zeros(n)
    KEVEC[1] = 1/2*transpose(vvec[:,1])*invM*vvec[:,1]
    HVEC = zeros(n)
    HVEC[1] = (1-λ)*U0VEC[1] + λ*UVEC[1] + KEVEC[1]
    for i = 2:n
        tempv = vvec[:,i-1] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        xvec[:,i] = xvec[:,i-1] .+ ϵ*invM*tempv
        U0VEC[i],grad0 = model.U0(xvec[:,i],requires_grad=true)
        UVEC[i],grad1  = model.U(xvec[:,i],requires_grad=true)
        vvec[:,i] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
        KEVEC[i] = 1/2*transpose(vvec[:,i])*invM*vvec[:,i]
        HVEC[i] = (1-λ)*U0VEC[i] + λ*UVEC[i] + KEVEC[i]
    end
    return xvec,vvec,U0VEC,UVEC,KEVEC,HVEC
end
function OneStepExplore(X,W,M,P,ϵ,model,λ,mass_mat)
    VMat = zeros(model.D,P,M)
    XMat = zeros(model.D,P,M)
    U0Mat = zeros(P,M)
    UMat  = zeros(P,M)
    KEMat = zeros(P,M)
    HMat  = zeros(P,M)
    A = vcat(fill.(1:(M*P),rand(Multinomial(M,W)))...)
    if mass_mat == "identity"
        invm = diagm(ones(model.D))
        m = diagm(ones(model.D))
    elseif mass_mat == "adaptive"
        Σ = cov(X,Weights(W))
        invm = diagm(diag(Σ))
        m    = inv(invm)
    end
    for n = 1:M
        XMat[:,:,n],VMat[:,:,n],U0Mat[:,n],UMat[:,n],KEMat[:,n],HMat[:,n] = ψ(X[:,A[n]],rand(MultivariateNormal(zeros(model.D),m)),P,invm,ϵ=ϵ,λ=λ,model=model)
    end
    return XMat,VMat,U0Mat,UMat,KEMat,HMat
end
function get_weights(u0,u,ke,prevλ,λ,M,P)
    log_weights = zeros(M*P)
    ind = 1
    for n = 1:M
        H0 = ke[1,n] + (1-prevλ)*u0[1,n] + prevλ*u[1,n]
        for k = 1:P
            log_weights[ind] =  - ke[k,n] - (1-λ)*u0[k,n] - λ*u[k,n] + H0
            ind += 1
        end
    end
    return log_weights
end
function getW(log_weights)
    MAX = findmax(log_weights)[1]
    W = exp.(log_weights.-MAX)/sum(exp.(log_weights.-MAX))
    return W
end
function ESS(log_weights)
    W = getW(log_weights)
    return 1/sum(W.^2)
end
function look_for_next_lambda(u0,u,ke,prevλ,α,M,P)
    res = optimize(tar->-ESS(get_weights(u0,u,ke,prevλ,tar,M,P)),prevλ,1.0)
    maxESS = -res.minimum
    f(lambda) = ESS(get_weights(u0,u,ke,prevλ,lambda,M,P)) - α*maxESS
    rt = find_zeros(f,prevλ,2.0)
    if length(rt) == 0
        return (1+prevλ)/2
    else
        return rt[1]
    end
end
function merge_array(x)
    D,P,M = size(x)
    out = zeros(D,M*P)
    ind = 1
    for n = 1:M
        for k = 1:P
            out[:,ind] = x[:,k,n]
            ind += 1
        end
    end
    return out 
end
function SMC(N,M;model,ϵ,α,mass_mat="adaptive",printl=false)
    X = Array{Matrix,1}(undef,0);push!(X,zeros(model.D,N))
    λ = zeros(1)
    logW = zeros(N,1)
    W = zeros(N,1)
    P = div(N,M)
    H = zeros(N,1)
    for i = 1:N
        X[1][:,i] = rand(model.initDist)
        v = randn(model.D)
        logW[i,1] = 0.0
        H[i,1] = model.U0(X[1][:,i],requires_grad=false) + 1/2*norm(v).^2
    end
    MAX = findmax(logW[:,1])[1]
    W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
    t = 1
    while λ[end] < 1.0
        t += 1
        x,v,u0,u,ke,h = OneStepExplore(X[t-1],W[:,t-1],M,P,ϵ,model,λ[t-1],mass_mat)
        if printl
            println("looking for new lambda ")
        end
        newλ = look_for_next_lambda(u0,u,ke,λ[t-1],α,M,P)
        if printl
            println("the new temperature at iteration $(t) is ",newλ)
        end
        if newλ >1.0
            push!(λ,1.0)
        else
            push!(λ,newλ)
        end
        push!(X,merge_array(x))
        logW = hcat(logW,get_weights(u0,u,ke,λ[t-1],λ[t],M,P))
        H = hcat(H,reshape(h,:,1))
        W = hcat(W,getW(logW[:,t]))
    end
    logNC = sum(log.(mean(exp.(logW),dims=1)))
    MM = sum(W[:,end].*mean(X[end],dims=1)[1,:])
    return (X=X,λ=λ,W=W,logW=logW,H=H,logNC=logNC,MM=MM)
end
end