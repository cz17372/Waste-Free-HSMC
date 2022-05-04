using Distributions, Plots, LinearAlgebra, Colors, StatsBase
theme(:ggplot2)
include("Models/sonar.jl")
include("src/WasteFree4.jl")
function ESS(X,H,T,P,M)
    W = zeros(P)
    Dist = zeros(P)
    for i = 1:M
        W[i] = exp.(-H[(i-1)*P+T] .+ H[(i-1)*P+1])
        Dist[i] = norm(X[(i-1)*P+T,:] .- X[(i-1)*P+1,:])^2
    end
    return sum((W/sum(W)) .* Dist)
end
function ESJD(R,M,P)
    T = length(R.X)-1
    out = zeros(T,P)
    for n = 1:T
        out[n,:] = ESS.(Ref(R.X[n+1]),Ref(R.H[:,n+1]),collect(1:P),Ref(P),Ref(M))
    end
    return out
end

N = 10000; M = 100; P = div(N,M)
include("src/WasteFree4.jl")
R = WasteFree.SMC(N,M,model=sonar,ϵ=0.2,α=0.5,method="full",mass_mat="identity",printl=true);
VEC02 = ESJD(R,M,P)
col = colormap("Blues",length(R.λ)-1);
plot(0.2*(1:P),VEC02[1,:],label="",color=col[1],xlabel="t",ylabel="Weighted ESJD",title="stepsize=0.2")
for n = 2:size(VEC02)[1]
    plot!(0.2*(1:P),VEC02[n,:],label="",color=col[n])
end
current()

function getmaxind(x)
    return findmax(x)[2]
end

ϵ2= mapslices(getmaxind,x,dims=2)[:,1] * 0.2 / P
λ = R.λ


include("src/WasteFree3.jl")
R = WasteFree.SMC(N,M,model=sonar,λ=λ,ϵ=ϵ,α=0.5,method="full",mass_mat="identity",printl=true);
function LogNC(R)
    return sum(log.(mean(exp.(R.logW),dims=1)[1,:]))
end
LogNC(R)

function tr(Mat)
    M,N = size(Mat)
    out = zeros(M,N)
    for m = 1:M
        out[m,:] = cumsum(Mat[m,:]) ./ (collect(1:N))
    end
    return out
end
col = colormap("Blues",length(R.λ)-1);
x = tr(VEC02)
plot(x[1,:],label="",color=col[1])

for n = 2:size(x)[1]
    plot!(x[n,:],label="",color=col[n])
end
current()