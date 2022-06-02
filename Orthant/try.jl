using Distributions,Plots,LinearAlgebra,StatsPlots, Roots,NPZ
theme(:ggplot2)
a = npzread("Orthant/a_rc150.npy")
M = Hermitian(npzread("Orthant/Sigma_rc150.npy"))
function exactHamltonian(x0,p0,t,M,invM)
    x1 = x0 * cos(t) .+ M*p0*sin(t)
    p1 = -invM*x0*sin(t) .+ p0*cos(t)
    return (x1,p1)
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
function find_exact_bounce_time(x0,p0,M,invM,lower=repeat([-Inf],length(x0)),upper=repeat([Inf],length(x0)))
    avec = x0
    bvec = M*p0
    min_lower_bound_hitting_time = findmin(solvetrig.(avec,bvec,lower))[1]
    min_upper_bound_hitting_time = findmin(solvetrig.(avec,bvec,upper))[1]
    return min(min_lower_bound_hitting_time,min_upper_bound_hitting_time)
end
function plottraj(x0,p0,t,M,invM,fig=nothing)
    ϵ = t/100
    traj = zeros(length(x0),100)
    pvec = zeros(length(x0),100)
    for n = 1:100
        traj[:,n],pvec[:,n] = exactHamltonian(x0,p0,n*ϵ,M,invM)
    end
    if isnothing(fig)
        plot(traj[1,:],traj[2,:],label="",color=:green,linewidth=5.0)
        scatter!([x0[1]],[x0[2]],label="",color=:yellow,markersize=5,markerstrokewidth=0.0)
        scatter!([traj[1,end]],[traj[2,end]],label="",color=:yellow,markersize=5,markerstrokewidth=0.0)
        current()
    else
        plot!(fig,traj[1,:],traj[2,:],label="",color=:green,linewidth=5.0)
        scatter!(fig,[x0[1]],[x0[2]],label="",color=:yellow,markersize=5,markerstrokewidth=0.0)
        scatter!(fig,[traj[1,end]],[traj[2,end]],label="",color=:yellow,markersize=5,markerstrokewidth=0.0)
        current()
    end
end
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
C(x) = Int64(prod(x .> a))
gradC(x) = normalize(x .<= a)
r(u,v)   = u - 2*dot(u,v)/norm(v)^2*v
x0 = truncatedGaussian(zeros(150),Σ,a)
p0 = rand(MultivariateNormal(zeros(150),inv(M)))
x1,p1 = exactHamltonian(x0,p0,0.05,M,inv(M))
C(x1)
rp0 = r(p1,gradC(x1))
x2,p2 = exactHamltonian(x1,rp0,0.05,M,inv(M)) 
C(x2)

M2 = M[1:2,1:2]
a  = [1.5,1.5]
true_x = zeros(2,10000)
d = MultivariateNormal(zeros(2),M2)
invD = MultivariateNormal(zeros(2),inv(M2))
for n = 1:10000
    true_x[:,n] = truncatedGaussian(zeros(2),M2,a)
end
scatter(true_x[1,:],true_x[2,:],label="",markersize=0.5,markerstrokewidth=0.0,color=:grey,xlim=(-4,4),ylim=(-4,4))

xvec = collect(-4:0.05:4)
yvec = collect(-4:0.05:4)
zvec = zeros(length(xvec),length(yvec))
for i = 1:length(xvec)
    for j = 1:length(yvec)
        zvec[i,j] = -logpdf(d,[xvec[i],yvec[j]])
    end
end

fig1 = heatmap(xvec,yvec,zvec);plot!([1.5,4.0],[1.5,1.5],label="",color=:grey,linewidth=2.0,linestyle=:dash)
plot!(fig1,[1.5,1.5],[1.5,4.0],label="",color=:grey,linewidth=2.0,linestyle=:dash)
x0 = truncatedGaussian(zeros(2),M2,a)
p0 = rand(invD)
@time first_hit = find_exact_bounce_time(x0,p0,M2,inv(M2),a)
x1,p1 = exactHamltonian(x0,p0,first_hit,M2,inv(M2))
plottraj(x0,p0,first_hit,M2,inv(M2),fig1)

lower = a
upper = [Inf,Inf]
M = M2
avec = x0
bvec = M*p0
min_lower_bound_hitting_time = findmin(solvetrig.(avec,bvec,lower))[1]
min_upper_bound_hitting_time = findmin(solvetrig.(avec,bvec,upper))[1]
f(t) = avec[2]*cos(t) + bvec[2]*sin(t) - a[2]
rt = find_zeros(f,0,2*pi)
t =0.6861658425186719


solvetrig.(avec,bvec,lower)
a = avec[2]
b = bvec[2]
c = 1.5
R = sqrt(a^2 + b^2)
φ = atan(a/b)
θ = asin(c/R)
θ - π

x = rand(2,2)
M = transpose(x)*x
L = cholesky(M).L
a = [1.5,1.5]

x0 = truncatedGaussian(1,zeros(2),M,a)
v0 = randn(2)
ϵ = 0.5
L*(x0+ϵ*v0)
L*x0
L*v0
