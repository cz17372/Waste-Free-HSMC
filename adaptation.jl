using Distributions, Plots, LinearAlgebra, Colors, StatsBase, LaTeXStrings
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
function SingleJumpVariance(X,W,P,M)
    D = zeros(M,P)
    JumpVar = zeros(M,P)
    for n = 1:M
        for i = 1:P
            D[n,i] = norm(X[(n-1)*P+i,:] .- X[(n-1)*P+1,:])^2
        end
        JumpVar[n,:] = cumsum(W[(n-1)*P+1:n*P] .* D[n,:]) ./ cumsum(W[(n-1)*P+1:n*P])
    end
    return mapslices(mean,JumpVar,dims=1)[1,:]
end
function JumpVariance(R,M,P)
    T = length(R.λ)-1
    out = zeros(T,P)
    for n = 1:T
        out[n,:] = SingleJumpVariance(R.X[n+1],R.W[:,n+1],P,M)
    end
    return out
end
function plotJumpVar(R,ss,M,P;col="Blues",fig=nothing)
    T = length(R.λ) -1
    c = colormap(col,T+10);
    out = JumpVariance(R,M,P)
    if isnothing(fig)
        plot(ss*collect(1:P),out[1,:],xlabel=L"\tau",ylabel="Jumping Variance",label="",color=c[11],framestyle=:box)
        for n = 2:T
            plot!(ss*collect(1:P),out[n,:],label="",color=c[n+10])
        end
        current()
    else
        for n = 1:T
            plot!(ss*collect(1:P),out[n,:],label="",color=c[n+10])
        end
        current()
    end
end
function LogNC(R)
    return sum(log.(mean(exp.(R.logW),dims=1)[1,:]))
end
getmaxidx(x) = findmax(x)[2]
function SingleTrajESS(W,M,P)
    res = zeros(M,P)
    for n = 1:M
        tempW = W[(n-1)*P+1:n*P]
        for t = 1:P
            res[n,t] = (1/sum((tempW[1:t] / sum(tempW[1:t])).^2))/t
        end
    end
    return mapslices(mean,res,dims=1)[1,:]
end
function TrajESS(R,M,P)
    T = length(R.λ) -1
    output = zeros(T,P)
    for n = 1:T
        output[n,:] = SingleTrajESS(R.W[:,n+1],M,P)
    end
    return output
end
function plotTrajESS(R,ss,M,P;col="Blues",fig=nothing)
    T = length(R.λ) -1
    c = colormap(col,T+10);
    out = TrajESS(R,M,P)
    if isnothing(fig)
        plot(ss*collect(1:P),out[1,:],xlabel="t",ylabel="%ESS",label="",color=c[1])
        for n=2:T
            if n == T
                plot!(ss*collect(1:P),out[n,:],label="eps=$(ss)",color=c[n])
            else
                plot!(ss*collect(1:P),out[n,:],label="",color=c[n])
            end
        end
        current()
    else
        for n=1:T
            if n == T
                plot!(fig,ss*collect(1:P),out[n,:],label="eps=$(ss)",color=c[n])
            else
                plot!(fig,ss*collect(1:P),out[n,:],label="",color=c[n])
            end
        end
        current()
    end
end
function CrazyIdea(R,M,P)
    T = length(R.λ)-1
    output = zeros(T,P)
    for n = 1:T
        output[n,:] = SingleTrajESS(R.W[:,n+1],M,P) .* SingleJumpVariance(R.X[n+1],R.W[:,n+1],P,M)
    end
    return output
end
function plotCI(R,ss,M,P;col="Blues",fig=nothing)
    T = length(R.λ) -1
    c = colormap(col,T+10);
    out = CrazyIdea(R,M,P)
    if isnothing(fig)
        plot(ss*collect(1:P),out[1,:],xlabel="t",ylabel="",label="",color=c[1])
        for n=2:T
            if n == T
                plot!(ss*collect(1:P),out[n,:],label="",color=c[n])
            else
                plot!(ss*collect(1:P),out[n,:],label="",color=c[n])
            end
        end
        current()
    else
        for n=1:T
            if n == T
                plot!(fig,ss*collect(1:P),out[n,:],label="",color=c[n])
            else
                plot!(fig,ss*collect(1:P),out[n,:],label="",color=c[n])
            end
        end
        current()
    end
end

using JLD2; @load "data.jld2" λ τ
include("src/WasteFree3.jl")
eps = 0.1; N = Int(120/eps*100); M = 100; P = div(N,M);R = WasteFree.SMC(N,M,model=sonar,λ=λ,ϵ=eps*ones(length(λ)-1),α=0.5,method="full",mass_mat="identity",printl=true);
fig2 = plotTrajESS(R,0.05,M,P)
plotTrajESS(R,eps,M,P,col="Grays",fig=fig2)

plotCI(R,eps,M,P)
plot(mapslices(getmaxidx,CrazyIdea(R,M,P),dims=2)[:,1] * eps)

fig1 = plotJumpVar(R,eps,M,P)