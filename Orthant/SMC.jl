using Distributions,LinearAlgebra,ProgressMeter
function reflect(u,v)
    return u .- 2*dot(u,v)*v
end
function solvetrig(a,b,c)
    R = sqrt(a^2 + b^2)
    if abs(c/R) > 1
        return Inf
    end
    φ = atan(b/a)
    θ = acos(c/R)
    if θ < -φ
        return 2*pi-θ+φ
    else
        return θ + φ
    end
end
function find_hitting_times(x0::Vector{Float64},v0::Vector{Float64},L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64})
    tx = L*x0; tv = L*v0
    ht_a,ia = findmin(solvetrig.(tx,tv,a))
    ht_b,ib = findmin(solvetrig.(tx,tv,b))
    if isinf(ht_a) & isinf(ht_b)
        return Inf,nothing
    elseif ht_a < ht_b
        return ht_a,ia
    else
        return ht_b,ib
    end
end
function ψexact(x0::Vector{Float64},v0::Vector{Float64},ϵ::Float64,L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64};keeptraj=false)
    d = length(x0)
    x = zeros(d,1);x[:,1] = x0
    v = zeros(d,1);v[:,1] = v0
    remaining_time = ϵ
    ht,i = find_hitting_times(x[:,end],v[:,end],L,a,b)
    nbounces = 0
    while ht < remaining_time
        nbounces += 1
        newx = x[:,end]*cos(ht) .+ v[:,end]*sin(ht)
        n = normalize(L[i,:])
        newv = reflect(-x[:,end]*sin(ht) .+ v[:,end]*cos(ht),n)
        x = hcat(x,newx)
        v = hcat(v,newv)
        remaining_time -= ht
        ht,i = find_hitting_times(x[:,end],v[:,end],L,a,b)
    end
    newx = x[:,end]*cos(remaining_time) .+ v[:,end]*sin(remaining_time)
    newv = -x[:,end]*sin(remaining_time) .+ v[:,end]*cos(remaining_time)
    x = hcat(x,newx)
    v = hcat(v,newv)
    if keeptraj
        return x,v,nbounces
    else
        return x[:,end],v[:,end],nbounces
    end
end
function LeapFrog(x0::Vector{Float64},v0::Vector{Float64},n::Int64,ϵ::Float64,L::Matrix{Float64},a::Vector{Float64},b::Vector{Float64};keeptraj=false)
    D = length(x0)
    outx = zeros(D,n+1);outx[:,1] = x0
    outv = zeros(D,n+1);outv[:,1] = v0
    nbounces = 0
    for i = 1:n
        outx[:,i+1],outv[:,i+1],nb = ψexact(outx[:,i],outv[:,i],ϵ,L,a,b)
        nbounces += nb
    end
    if keeptraj
        return outx,outv,nbounces/(n-1)
    else
        return outx[:,end],outv[:,end],nbounces/(n-1)
    end
end
function generate_trajectory(x0,n,ϵ,L,a,b)
    d = length(x0)+1
    newx,nc = conditionalOrthant(x0,L,a,b)
    v0 = randn(length(newx))
    x,v,nb = LeapFrog(newx,v0,n-1,ϵ,L[1:d,1:d],a[1:d],b[1:d],keeptraj=true)
    lw = log(nc)*ones(n)
    return x,lw,nb
end
function get_particles(X,W,M,P,ϵ,L,a,b)
    D,N = size(X)
    D +=1
    newX = zeros(D,N)
    logW = zeros(N)
    A = vcat(fill.(1:N,rand(Multinomial(M,W)))...)
    nbvec = zeros(M)
    for n = 0:(M-1)
        newX[:,n*P+1:(n+1)*P],logW[n*P+1:(n+1)*P],nbvec[n+1] = generate_trajectory(X[:,A[n+1]],P,ϵ,L,a,b)
    end
    return newX,logW,mean(nbvec)
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
function SMC(N,M,ϵ0,L,a,b,niter)
    P = div(N,M)
    X = zeros(niter,N)
    logW = zeros(N,niter)
    W = zeros(N,niter)
    for n = 1:N
        X[1:1,n],_ = conditionalOrthant(zeros(0),L,a,b)
        logW[n,1] = 0.0
    end
    MAX = findmax(logW[:,1])[1]
    W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
    LogNC = 0.0
    ϵ = ϵ0
    for t = 2:niter
        println(t)
        X[1:t,:],logW[:,t],nb = get_particles(X[1:(t-1),:],W[:,t-1],M,P,ϵ,L,a,b)
        if nb > 5
            ϵ = ϵ*0.9
        end
        LogNC += log(mean(exp.(logW[:,t])))
        println(LogNC,"No. Bounces = ",nb,"ϵ = ",ϵ)
        MAX = findmax(logW[:,t])[1]
        W[:,t] = exp.(logW[:,t] .- MAX)/sum(exp.(logW[:,t] .- MAX))
        #println("ESS = ",1/sum(W[:,t].^2))
    end
    return X,logW,W,LogNC
end