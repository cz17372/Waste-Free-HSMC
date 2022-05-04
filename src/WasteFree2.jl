module WasteFree
using Distributions, Roots, LinearAlgebra, StatsBase, Optim
using ForwardDiff:gradient
export SMC
function ψ(x0,v0,n,invM;ϵ,model,λ)
    forwarditer = div(n,2)
    backwarditer = n - forwarditer
    fxvec = zeros(forwarditer+1,model.D)
    fvvec = zeros(forwarditer+1,model.D)
    fxvec[1,:] = x0
    fvvec[1,:] = v0
    fU0VEC = zeros(forwarditer+1)
    fUVEC  = zeros(forwarditer+1)
    fU0VEC[1],grad0 = model.U0(x0,requires_grad=true)
    fUVEC[1],grad1  = model.U(x0,requires_grad=true)
    fKEVEC = zeros(forwarditer+1)
    fKEVEC[1] = 1/2*transpose(fvvec[1,:])*invM*fvvec[1,:]
    for i = 1:forwarditer
        tempv = fvvec[i,:] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        fxvec[i+1,:] = fxvec[i,:] .+ ϵ*invM*tempv
        fU0VEC[i+1],grad0 = model.U0(fxvec[i+1,:],requires_grad=true)
        fUVEC[i+1],grad1  = model.U(fxvec[i+1,:],requires_grad=true)
        fvvec[i+1,:] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
        fKEVEC[i+1] = 1/2*transpose(fvvec[i+1,:])*invM*fvvec[i+1,:]
    end
    bxvec = zeros(backwarditer+1,model.D)
    bvvec = zeros(backwarditer+1,model.D)
    bxvec[1,:] = x0
    bvvec[1,:] = -v0
    bU0VEC = zeros(backwarditer+1)
    bUVEC  = zeros(backwarditer+1)
    bU0VEC[1],grad0 = model.U0(x0,requires_grad=true)
    bUVEC[1],grad1  = model.U(x0,requires_grad=true)
    bKEVEC = zeros(backwarditer+1)
    bKEVEC[1] = 1/2*transpose(bvvec[1,:])*invM*bvvec[1,:]
    for i = 1:backwarditer
        tempv = bvvec[i,:] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        bxvec[i+1,:] = bxvec[i,:] .+ ϵ*invM*tempv
        bU0VEC[i+1],grad0 = model.U0(bxvec[i+1,:],requires_grad=true)
        bUVEC[i+1],grad1  = model.U(bxvec[i+1,:],requires_grad=true)
        bvvec[i+1,:] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
        bKEVEC[i+1] = 1/2*transpose(bvvec[i+1,:])*invM*bvvec[i+1,:]
    end
    return [fxvec;bxvec[2:end,:]], [fvvec;bvvec[2:end,:]], [fU0VEC;bU0VEC[2:end]], [fUVEC;bUVEC[2:end]], [fKEVEC;bKEVEC[2:end]]
end
function OneStepExplore(X,W,M,P;ϵ,model,λ,mass_mat)
    VMat = zeros(P,model.D,M)
    XMat = zeros(P,model.D,M)
    U0Mat = zeros(P,M)
    UMat  = zeros(P,M)
    KEMat = zeros(P,M)
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
        x0 = X[A[n],:]
        v0 = rand(MultivariateNormal(zeros(model.D),m))
        XMat[:,:,n],VMat[:,:,n],U0Mat[:,n],UMat[:,n],KEMat[:,n] = ψ(x0,v0,P-1,invm,ϵ=ϵ,λ=λ,model=model)
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
function get_particles(X,W,M,P;ϵ,model,λ,method,mass_mat)
    if method == "full"
        return OneStepExplore(X,W,M,P,ϵ=ϵ,model=model,λ = λ,mass_mat=mass_mat)
    elseif method == "sequential"
        x,v,u0,u,ke = OneStepExplore(X,W,M,P,ϵ=ϵ,model=model,λ = λ,mass_mat=mass_mat)
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
        x,v,u0,u,ke = OneStepExplore(X,W,M,P,ϵ=ϵ,model=model,λ = λ,mass_mat=mass_mat)
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
    if method == "full"
        res = optimize(tar->-ESS(get_weights(x,v,u0,u,ke,prevλ,tar,method=method)),prevλ,1.0)
        maxESS = -res.minimum
        #println("The max ESS achieves at temp = $(Optim.minimizer(res))")
    else
        maxESS = ESS(get_weights(x,v,u0,u,ke,prevλ,prevλ,method=method))
    end
    #println(maxESS)
    f(lambda) = ESS(get_weights(x,v,u0,u,ke,prevλ,lambda,method=method)) - α*maxESS
    rt = find_zeros(f,prevλ,2.0)
    return rt[1]
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
function SMC(N,M;model,ϵ,α,method,mass_mat="adaptive",printl=false)
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
    temp_time = 0.0
    exp_time  = 0.0
    while λ[end] < 1.0
        t += 1
        t1 = @timed x,v,u0,u,ke = get_particles(X[t-1],W[:,t-1],M,P,ϵ=ϵ,model=model,λ=λ[t-1],method=method,mass_mat=mass_mat)
        exp_time += t1.time - t1.gctime
        if printl
            println("looking for new lambda ")
        end
        t2 = @timed newλ = look_for_next_lambda(x,v,u0,u,ke,λ[t-1],α,method=method)
        temp_time += t2.time - t2.gctime
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
    return (X=X,λ=λ,W=W,logW=logW,explore_time = exp_time,temp_time=temp_time)
end
end
