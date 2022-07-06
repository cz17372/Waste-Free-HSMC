module chopin
using Distributions, Roots, LinearAlgebra, StatsBase, Optim
function MH(x0,niter,Σ,model,λ)
    output = zeros(model.D,niter)
    output[:,1] = x0
    U0VEC = zeros(niter)
    UVEC = zeros(niter)
    U0VEC[1] = model.U0(x0,requires_grad=false)
    UVEC[1]  = model.U(x0,requires_grad=false)
    prevH = (1-λ)*U0VEC[1] + λ*UVEC[1]
    for n = 2:niter
        newx = rand(MultivariateNormal(output[:,n-1],2.38^2/model.D*Σ))
        newu0 = model.U0(newx,requires_grad=false)
        newu  = model.U(newx,requires_grad=false)
        newH = (1-λ)*newu0 + λ*newu
        if log(rand()) < -newH + prevH
            output[:,n] = newx
            U0VEC[n] = newu0
            UVEC[n] = newu
            prevH = newH
        else
            output[:,n] = output[:,n-1]
            U0VEC[n] = U0VEC[n-1]
            UVEC[n] = UVEC[n-1]
        end
    end
    return output, U0VEC, UVEC
end
function ChopinOneStepExplore(X,W,M,P;model,λ)
    XMat = zeros(model.D,P,M)
    U0Mat = zeros(P,M)
    UMat  = zeros(P,M)
    A = vcat(fill.(1:(M*P),rand(Multinomial(M,W)))...)
    Σ = cov(Matrix(X'),Weights(W))
    for n = 1:M
        x0 = X[:,A[n]]
        XMat[:,:,n],U0Mat[:,n],UMat[:,n] = MH(x0,P,Σ,model,λ)
    end
    return XMat,U0Mat,UMat
end
function ESS(log_weights)
    W = getW(log_weights)
    return 1/sum(W.^2)
end
function getW(log_weights)
    MAX = findmax(log_weights)[1]
    W = exp.(log_weights.-MAX)/sum(exp.(log_weights.-MAX))
    return W
end
function get_weights(u0,u,prevλ,λ,M,P)
    N = P*M
    log_weights = zeros(N)
    ind = 1
    for n = 1:M
        for k = 1:P
            log_weights[ind] = (1-prevλ)*u0[k,n] + prevλ*u[k,n] - (1-λ)*u0[k,n] - λ*u[k,n]
            ind += 1
        end
    end
    return log_weights
end
function look_for_next_lambda(u0,u,prevλ,α,M,P)
    maxESS = ESS(get_weights(u0,u,prevλ,prevλ,M,P))
    f(lambda) = ESS(get_weights(u0,u,prevλ,lambda,M,P)) - α*maxESS
    rt = find_zeros(f,prevλ,2.0)
    return rt[1]
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
function SMC(N,M;model,α,printl=false)
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
        x,u0,u = ChopinOneStepExplore(X[t-1],W[:,t-1],M,P,model=model,λ=λ[t-1])
        if printl
            println("looking for new lambda ")
        end
        newλ = look_for_next_lambda(u0,u,λ[t-1],α,M,P)
        if printl
            println("the new temperature at iteration $(t) is ",newλ)
        end
        if newλ >1.0
            push!(λ,1.0)
        else
            push!(λ,newλ)
        end
        push!(X,merge_array(x))
        logW = hcat(logW,get_weights(u0,u,λ[t-1],λ[t],M,P))
        W = hcat(W,getW(logW[:,t]))
    end
    logNC = sum(log.(mean(exp.(logW),dims=1)))
    MM = sum(W[:,end].*mean(X[end],dims=1)[1,:])
    return (X=X,λ=λ,W=W,logW=logW,logNC=logNC,MM=MM)
end



end