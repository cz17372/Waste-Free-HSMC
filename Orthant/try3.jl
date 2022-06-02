using Distributions,Plots,LinearAlgebra,StatsPlots, Roots,NPZ,ProgressMeter
H(x,v) = 1/2*transpose(x)*x + 1/2*transpose(v)*v
function reflect(u,v)
    return u .- 2*dot(u,v)*v
end
function find_hitting_times(x0::Vector{Float64},v0::Vector{Float64},L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64})
    D = length(x0)
    res = Inf
    idx = nothing
    tx0 = L*x0; tv0 = L*v0
    for n = 1:D
        lht = (a[n]-tx0[n])/tv0[n]
        if (lht>1e-10) & (lht<res)
            res = lht
            idx = n
        end
        uht = (b[n] - tx0[n])/tv0[n]
        if (uht>1e-10) & (uht<res)
            res = uht
            idx = n
        end
    end
    return res,idx
end
function full_step(x0::Vector{Float64},v0::Vector{Float64},L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64},ϵ::Float64;keeptraj=false)
    d = length(x0)
    x = zeros(d,1);x[:,1] = x0
    v = zeros(d,1);v[:,1] = v0
    remaining_time = ϵ
    ht,i = find_hitting_times(x[:,end],v[:,end],L,a,b)
    while ht < remaining_time
        newx = x[:,end] .+ ht*v[:,end]
        n = normalize(L[i,:])
        newv = reflect(v[:,end],n)
        x = hcat(x,newx)
        v = hcat(v,newv)
        remaining_time -= ht
        ht,i = find_hitting_times(x[:,end],v[:,end],L,a,b)
    end
    newx = x[:,end] .+ remaining_time*v[:,end]
    newv = v[:,end]
    x = hcat(x,newx)
    v = hcat(v,newv)
    if keeptraj
        return x,v
    else
        return x[:,end],v[:,end]
    end
end
function ψ(x0::Vector{Float64},v0::Vector{Float64},ϵ::Float64,L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64})
    # Perform a half leap-frog step 
    tempv = v0 .- ϵ/2*x0
    newx,tempv = full_step(x0,tempv,L,a,b,ϵ)
    newv  = tempv .- ϵ/2*newx
    return newx,newv
end
function LeapFrog(x0::Vector{Float64},v0::Vector{Float64},n::Int64,ϵ::Float64,L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64};keeptraj=false)
    D = length(x0)
    outx = zeros(D,n+1);outx[:,1] = x0
    outv = zeros(D,n+1);outv[:,1] = v0
    for i = 1:n
        outx[:,i+1],outv[:,i+1] = ψ(outx[:,i],outv[:,i],ϵ,L,a,b)
    end
    if keeptraj
        return outx,outv
    else
        return outx[:,end],outv[:,end]
    end
end
function truncatedGaussian(N,μ,Σ,lower=nothing,upper=nothing)
    d = length(μ)
    L = Matrix(cholesky(Σ).L)
    if isnothing(lower)
        a = repeat([-Inf],d)
    else
        a = lower
    end
    if isnothing(upper)
        b = repeat([Inf],d)
    else
        b = upper
    end
    x = zeros(d,N)
    for i = 1:N
        x[1,i] = rand(truncated(Normal(0,1),a[1]/L[1,1],b[1]/L[1,1]))
        for n = 2:d
            l = (a[n] - sum(L[n,1:n-1].*x[1:n-1,i]))/L[n,n]
            u = (b[n] - sum(L[n,1:n-1].*x[1:n-1,i]))/L[n,n]
            x[n,i] = rand(truncated(Normal(0,1),l,u))
        end
    end
    return x
end
function HMC(x0::Vector{Float64},N::Int64,ϵ::Float64,nint::Int64,L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64})
    D = length(x0)
    outx = zeros(D,N+1);outx[:,1] = x0
    outv = zeros(D,N);
    acc = 0
    @showprogress 1 "Computing..." for n = 1:N
        #println(n)
        v0 = randn(D)
        outv[:,n] = v0
        newx,newv = LeapFrog(outx[:,n],v0,nint,ϵ,L,a,b)
        if log(rand()) < -H(newx,newv)+H(outx[:,n],v0)
            outx[:,n+1] = newx
            acc += 1
        else
            outx[:,n+1] = outx[:,n]
        end
    end
    return outx,outv,acc/N
end
function HMCexact(x0::Vector{Float64},N::Int64,ϵ::Float64,L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64})
    D = length(x0)
    outx = zeros(D,N+1);outx[:,1] = x0
    outv = zeros(D,N);
    bouncvec = zeros(N)
    @showprogress 1 "Computing..." for n = 1:N
        #println(n)
        v0 = randn(D)
        outv[:,n] = v0
        outx[:,n+1],_,bouncvec[n] = ψexact(outx[:,n],v0,ϵ,L,a,b)
    end
    return outx,outv,mean(bouncvec)
end
function generate_trajectory(x0,n,ϵ,L,a,b)
    d = length(x0)+1
    newx,nc = conditionalOrthant(x0,L,a,b)
    v0 = randn(length(newx))
    x,v = LeapFrog(newx,v0,n-1,ϵ,L[1:d,1:d],a[1:d],b[1:d],keeptraj=true)
    h  = (mapslices(norm2,x,dims=1)/2 .+ mapslices(norm2,v,dims=1)/2)[1,:]
    lw = -h .+ h[1] .+ log(nc)
    return x,lw
end
function get_particles(X,W,M,P,ϵ,L,a,b)
    D,N = size(X)
    D +=1
    newX = zeros(D,N)
    logW = zeros(N)
    A = vcat(fill.(1:N,rand(Multinomial(M,W)))...)
    for n = 0:(M-1)
        newX[:,n*P+1:(n+1)*P],logW[n*P+1:(n+1)*P] = generate_trajectory(X[:,A[n+1]],P,ϵ,L,a,b)
    end
    return newX,logW
end
norm2(x) = dot(x,x)
function conditionalOrthant(x0,L,a,b)
    n = length(x0)+1
    l = (a[n] - dot(L[n,(1:n-1)],x0))/L[n,n]
    u = (b[n] - dot(L[n,(1:n-1)],x0))/L[n,n]
    ext_x = rand(truncated(Normal(0,1),l,u))
    NC    = cdf(Normal(0,1),u) - cdf(Normal(0,1),l)
    return [x0;[ext_x]],NC
end
function SMC(N,M,ϵ,L,a,b,niter)
    P = div(N,M)
    X = Vector{Matrix}(undef,niter)
    X[1] = zeros(1,N)
    logW = zeros(N,niter)
    W = zeros(N,niter)
    for n = 1:N
        X[1][:,n],_ = conditionalOrthant(zeros(0),L,a,b)
        logW[n,1] = 0.0
    end
    MAX = findmax(logW[:,1])[1]
    W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
    LogNC = 0.0
    for t = 2:niter
        println(t)
        X[t],logW[:,t] = get_particles(X[t-1],W[:,t-1],M,P,ϵ,L,a,b)
        LogNC += log(mean(exp.(logW[:,t])))
        println(LogNC)
        MAX = findmax(logW[:,t])[1]
        W[:,t] = exp.(logW[:,t] .- MAX)/sum(exp.(logW[:,t] .- MAX))
        println("ESS = ",1/sum(W[:,t].^2))
    end
    return X,logW,W,LogNC
end
