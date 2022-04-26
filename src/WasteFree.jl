module WasteFree
using Distributions, Roots, LinearAlgebra, StatsBase
using ForwardDiff:gradient
export SMC
function ψ(x0,v0,n;ϵ,model,λ)
    xvec = zeros(n+1,model.D)
    vvec = zeros(n+1,model.D)
    xvec[1,:] = x0
    vvec[1,:] = v0
    U0VEC = zeros(n+1)
    UVEC  = zeros(n+1)
    U0VEC[1],grad0 = model.U0(x0,requires_grad=true)
    UVEC[1],grad1  = model.U(x0,requires_grad=true)
    KEVEC = zeros(n+1)
    KEVEC[1] = 1/2*norm(vvec[1,:])^2
    for i = 1:n
        tempv = vvec[i,:] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        xvec[i+1,:] = xvec[i,:] .+ ϵ*tempv
        U0VEC[i+1],grad0 = model.U0(xvec[i+1,:],requires_grad=true)
        UVEC[i+1],grad1  = model.U(xvec[i+1,:],requires_grad=true)
        vvec[i+1,:] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
        KEVEC[i+1] = 1/2*norm(vvec[i+1,:])^2
    end
    return xvec,vvec,U0VEC,UVEC,KEVEC
end
function OneStepExplore(X,W,M,P;ϵ,model,λ)
    VMat = zeros(P,model.D,M)
    XMat = zeros(P,model.D,M)
    U0Mat = zeros(P,M)
    UMat  = zeros(P,M)
    KEMat = zeros(P,M)
    A = vcat(fill.(1:(M*P),rand(Multinomial(M,W)))...)
    for n = 1:M
        x0 = X[A[n],:]
        v0 = randn(model.D)
        XMat[:,:,n],VMat[:,:,n],U0Mat[:,n],UMat[:,n],KEMat[:,n] = ψ(x0,v0,P-1,ϵ=ϵ,λ=λ,model=model)
    end
    return XMat,VMat,U0Mat,UMat,KEMat
end
function MH(x0,niter,Σ,model,λ)
    D = length(x0)
    output = zeros(niter,D)
    output[1,:] = x0
    U0VEC = zeros(niter)
    UVEC = zeros(niter)
    U0VEC[1] = model.U0(x0,requires_grad=false)
    UVEC[1]  = model.U(x0,requires_grad=false)

    prevH = (1-λ)*U0VEC[1] + λ*UVEC[1]
    for n = 2:niter
        newx = rand(MultivariateNormal(output[n-1,:],2.38^2/model.D*Σ))
        newu0 = model.U0(newx,requires_grad=false)
        newu  = model.U(newx,requires_grad=false)
        newH = (1-λ)*newu0 + λ*newu
        if log(rand()) < -newH + prevH
            output[n,:] = newx
            U0VEC[n] = newu0
            UVEC[n] = newu
            prevH = newH
        else
            output[n,:] = output[n-1,:]
            U0VEC[n] = U0VEC[n-1]
            UVEC[n] = UVEC[n-1]
        end
    end
    return output, U0VEC, UVEC
end
function ChopinOneStepExplore(X,W,M,P;ϵ,model,λ)
    XMat = zeros(P,model.D,M)
    U0Mat = zeros(P,M)
    UMat  = zeros(P,M)
    A = vcat(fill.(1:(M*P),rand(Multinomial(M,W)))...)
    Σ = cov(X,Weights(W))
    for n = 1:M
        x0 = X[A[n],:]
        XMat[:,:,n],U0Mat[:,n],UMat[:,n] = MH(x0,P,Σ,model,λ)
    end
    return XMat,nothing,U0Mat,UMat,nothing
end
function get_particles(X,W,M,P;ϵ,model,λ,method)
    if method == "full"
        return OneStepExplore(X,W,M,P,ϵ=ϵ,model=model,λ = λ)
    elseif method == "sequential"
        x,v,u0,u,ke = OneStepExplore(X,W,M,P,ϵ=ϵ,model=model,λ = λ)
        newx = zeros(P,model.D,M)
        newv = zeros(P,model.D,M)
        newu0= zeros(P,M)
        newu = zeros(P,M)
        newke= zeros(P,M)
        for n = 1:M
            current_ind = 1
            current_H = (1-λ)*u0[1,n] + λ*u[1,n] + ke[1,n]
            newx[1,:,n] = x[1,:,n]; newv[1,:,n] = v[1,:,n]; newu0[1,n] = u0[1,n]; newu[1,n] = u[1,n]; newke[1,n] = ke[1,n]
            for k = 2:P
                newH = (1-λ)*u0[current_ind+1,n] + λ*u[current_ind+1,n] + ke[current_ind+1,n]
                if log(rand()) < -newH + current_H
                    newx[k,:,n] = x[current_ind+1,:,n]
                    newv[k,:,n] = v[current_ind+1,:,n]
                    current_H = newH
                    newu0[k,n] = u0[current_ind+1,n]
                    newu[k,n]  = u[current_ind+1,n]
                    newke[k,n] = ke[current_ind+1,n]
                    current_ind+=1
                else
                    newx[k,:,n] = x[current_ind,:,n]
                    newv[k,:,n] = v[current_ind,:,n]
                    newu0[k,n] = u0[current_ind,n]
                    newu[k,n]  = u[current_ind,n]
                    newke[k,n] = ke[current_ind,n]
                end
            end
        end
        return newx,newv,newu0,newu,newke
    elseif method == "independent"
        x,v,u0,u,ke = OneStepExplore(X,W,M,P,ϵ=ϵ,model=model,λ = λ)
        for n = 1:M
            H0 = (1-λ)*u0[1,n] + λ*u[1,n] + ke[1,n]
            for k = 2:P
                H1 = (1-λ)*u0[k,n] + λ*u[k,n] + ke[k,n]
                if log(rand()) > -H1+H0
                    x[k,:,n] = x[1,:,n]
                    v[k,:,n] = v[1,:,n]
                    u0[k,n]  = u0[1,n]
                    u[k,n]   = u[1,n]
                    ke[k,n]  = ke[1,n]
                end
            end
        end
        return x,v,u0,u,ke
    elseif method == "chopin"
        return ChopinOneStepExplore(X,W,M,P,ϵ=ϵ,model=model,λ=λ)
    end
end
function get_weights(x,v,u0,u,ke,prevλ,λ;method)
    P,_,M = size(x)
    N = P*M
    log_weights = zeros(N)
    ind = 1
    if method == "full"
        for n = 1:M
            H0 = ke[1,n] + (1-prevλ)*u0[1,n] + prevλ*u[1,n]
            for k = 1:P
                log_weights[ind] =  - ke[k,n] - (1-λ)*u0[k,n] - λ*u[k,n] + H0
                ind += 1
            end
        end
        return log_weights
    else
        for n = 1:M
            for k = 1:P
                log_weights[ind] = (1-prevλ)*u0[k,n] + prevλ*u[k,n] - (1-λ)*u0[k,n] - λ*u[k,n]
                ind += 1
            end
        end
        return log_weights
    end  
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
function look_for_next_lambda(x,v,u0,u,ke,prevλ,α;method)
    P,_,M = size(x)
    f(lambda) = ESS(get_weights(x,v,u0,u,ke,prevλ,lambda,method=method)) - α*M*P
    a = prevλ
    b = prevλ+0.1
    while f(a)*f(b) >= 0
        b += 0.1
    end
    return find_zero(f,(a,b),Bisection())
end
function merge_array(x)
    P,D,M = size(x)
    out = zeros(P*M,D)
    ind = 1
    for n = 1:M
        for k = 1:P
            out[ind,:] = x[k,:,n]
            ind += 1
        end
    end
    return out 
end
function SMC(N,M;model,ϵ,α,method,printl=false)
    X = Array{Matrix,1}(undef,0);push!(X,zeros(N,model.D))
    λ = zeros(1)
    logW = zeros(N,1)
    W = zeros(N,1)
    P = div(N,M)
    for i = 1:N
        X[1][i,:] = rand(model.initDist)
        v = randn(model.D)
        logW[i,1] = 0.0
    end
    MAX = findmax(logW[:,1])[1]
    W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
    t = 1
    while λ[end] < 1.0
        t += 1
        x,v,u0,u,ke = get_particles(X[t-1],W[:,t-1],M,P,ϵ=ϵ,model=model,λ=λ[t-1],method=method)
        if printl
            println("looking for new lambda ")
        end
        newλ = look_for_next_lambda(x,v,u0,u,ke,λ[t-1],α,method=method)
        if printl
            println("the new temperature at iteration $(t) is ",newλ)
        end
        if newλ >1.0
            push!(λ,1.0)
        else
            push!(λ,newλ)
        end
        push!(X,merge_array(x))
        logW = hcat(logW,get_weights(x,v,u0,u,ke,λ[t-1],λ[t],method=method))
        W = hcat(W,getW(logW[:,t]))
    end
    return (X=X,λ=λ,W=W,logW=logW)
end
end