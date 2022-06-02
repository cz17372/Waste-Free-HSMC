using Distributions,Plots,LinearAlgebra,StatsPlots, Roots,NPZ
theme(:ggplot2)
a = npzread("Orthant/a_rc150.npy")
M = Hermitian(npzread("Orthant/Sigma_rc150.npy"))
d = 150
a = a[1:d]
M = M[1:d,1:d]
invM = inv(M)
L = Matrix(cholesky(M).L)
function truncatedGaussian(μ,Σ,lower=nothing,upper=nothing)
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
    x = zeros(d)
    x[1] = rand(truncated(Normal(0,1),a[1]/L[1,1],b[1]/L[1,1]))
    for n = 2:d
        l = (a[n] - sum(L[n,1:n-1].*x[1:n-1]))/L[n,n]
        u = (b[n] - sum(L[n,1:n-1].*x[1:n-1]))/L[n,n]
        x[n] = rand(truncated(Normal(0,1),l,u))
    end
    return L*x
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
    return L*x
end
function hitting_time(x0,v0,a)
    d = length(x0)
    ht = Inf
    for n = 1:d
        if v0[n] < 0
            bt = (x0[n]-a[n])/-v0[n]
            if bt < ht
                ht = bt
            end
        end
    end
    return ht
end
function reflect(u,v)
    return u .- 2*dot(u,v)/norm(v)^2*v
end
"""
function full_step(x0,v0,ϵ,a)
    d = length(x0)
    x = zeros(d,1);x[:,1] = x0
    v = zeros(d,1);v[:,1] = v0
    remaining_time = ϵ
    ht = hitting_time(x[:,end],v[:,end],a)
    while  ht < remaining_time
        newx = x[:,end] .+ ht*v[:,end]
        grad = normalize(newx .<= a)
        newv = reflect(v[:,end],grad)
        x = hcat(x,newx)
        v = hcat(v,newv)
        remaining_time -= ht
        ht = hitting_time(x[:,end],v[:,end],a)
    end
    newx = x[:,end] .+ remaining_time*v[:,end]
    newv = -v[:,end]
    x = hcat(x,newx)
    v = hcat(v,newv)
    return x,v
end
"""
function full_step(x0,v0,ϵ,a)
    remaining_time = ϵ
    oldx = x0
    oldv = v0
    ht = hitting_time(oldx,oldv,a)
    while  ht < remaining_time
        oldx = oldx .+ ht*oldv
        grad = normalize(oldx .<= a)
        oldv = reflect(oldv,grad)
        remaining_time -= ht
        ht = hitting_time(oldx,oldv,a)
    end
    oldx = oldx .+ remaining_time*oldv
    return oldx,oldv
end
function leapfrog(x0,v0,ϵ,gradU,a)
    tempv = v0 .- ϵ/2*gradU(x0)
    newx,tempv = full_step(x0,tempv,ϵ,a)
    newv = tempv .- ϵ/2*gradU(newx)
    return newx,newv
end
gradU(x) = invM*x
function ψ(x0,v0,ϵ,L,gradU,a;keeptraj=false)
    D = length(x0)
    x = zeros(D,L+1)
    v = zeros(D,L+1)
    x[:,1] = x0
    v[:,1] = v0
    for n = 2:(L+1)
        x[:,n],v[:,n] = leapfrog(x[:,n-1],v[:,n-1],ϵ,gradU,a)
    end
    if keeptraj
        return x,v
    else
        return x[:,end],v[:,end]
    end
end
H(x,v) = 1/2*transpose(x)*invM*x + 1/2*transpose(v)*v
function HMC(x0,N,ϵ,L,gradU,a)
    D = length(x0)
    output = zeros(D,N+1)
    output[:,1] = x0
    acc = 0
    for n = 1:N
        println(n)
        v0 = randn(D)
        newx,newv = ψ(output[:,n],v0,ϵ,L,gradU,a)
        log_alpha = -H(newx,newv) + H(output[:,n],v0)
        if log(rand()) < log_alpha
            output[:,n+1] = newx
            acc += 1
        else
            output[:,n+1] = output[:,n]
        end
    end
    return output,acc/N
end
truex = truncatedGaussian(50000,zeros(d),M,a)

x0 = truncatedGaussian(1,zeros(d),M,a)
v0 = randn(d)

x,v = ψ(x0,v0,0.001,1200,gradU,a,keeptraj=true)
x2,v2 = ψ(x[:,end],-v[:,end],0.2,20,gradU,a,keeptraj=true)
plot(x[1,:],x[2,:],label="")
plot!(x2[1,:],x2[2,:],label="")
R,acc = HMC(x0,50000,0.01,20,gradU,a)
RR = inv(L)*R
mean(mapslices(mean,RR,dims=1)[1,:])
k=9;density(R[k,:],label="HMC");density!(truex[k,:],label="True density")

k = 8;plot(truex[k,:]);plot!(R[k,:])


mean(mapslices(mean,output,dims=1)[1,:])
mean(mapslices(mean,R,dims=1)[1,:])

R = truncatedGaussian(1000,zeros(150),M,a)

x = rand(2)
M = x * transpose(x)
L = Matrix(cholesky(M).L)