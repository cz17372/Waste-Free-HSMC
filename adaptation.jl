using Distributions, Plots, LinearAlgebra, Colors, StatsBase
theme(:ggplot2)
include("Models/sonar.jl")
function SingleESJD(X,H,T,P,M)
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
        out[n,:] = SingleESJD.(Ref(R.X[n+1]),Ref(R.H[:,n+1]),collect(1:P),Ref(P),Ref(M))
    end
    return out
end
function SingleJumpVariance(X,H,P,M)
    W = zeros(M,P)
    D = zeros(M,P)
    JumpVar = zeros(M,P)
    for n = 1:M
        W[n,:] = exp.(H[(n-1)*P+1:n*P] .- H[(n-1)*P+1])
        for i = 1:P
            D[n,i] = norm(X[(n-1)*P+i,:] .- X[(n-1)*P+1,:])^2
        end
        JumpVar[n,:] = cumsum(W[n,:] .* D[n,:]) ./ (cumsum(W[n,:]))
    end
    return mapslices(mean,JumpVar,dims=1)[1,:]
end
function JumpVariance(R,M,P)
    T = length(R.λ)-1
    out = zeros(T,P)
    for n = 1:T
        out[n,:] = SingleJumpVariance(R.X[n+1],R.H[:,n+1],P,M)
    end
    return out
end
function plotJumpVar(R,ss,M,P;col="Blues",fig=nothing)
    T = length(R.λ) -1
    c = colormap(col,T);
    out = JumpVariance(R,M,P)
    if isnothing(fig)
        plot(ss*collect(1:P),out[1,:],xlabel="tau",ylabel="Jump Variance",label="",color=c[1])
        for n = 2:T
            plot!(ss*collect(1:P),out[n,:],xlabel="tau",ylabel="Jump Variance",label="",color=c[n])
        end
        current()
    else
        for n = 1:T
            plot!(ss*collect(1:P),out[n,:],xlabel="tau",ylabel="Jump Variance",label="",color=c[n])
        end
        current()
    end
end
function LogNC(R)
    return sum(log.(mean(exp.(R.logW),dims=1)[1,:]))
end



N = 30000; M = 100; P = div(N,M)
include("src/WasteFree4.jl")
R = WasteFree.SMC(N,M,model=sonar,ϵ=0.2,α=0.5,method="independent",mass_mat="identity",printl=true);
fig1 = plotJumpVar(R,0.4,M,P)
plotJumpVar(R,0.3,M,P,fig=fig1,col="Reds")
VEC02 = ESJD(R,M,P)
jjj,D,W = JumpVariance(R.X[20],R.H[:,20],P,M)


JV = JumpVariance(R,M,P)

plot(JV[end-8,:])