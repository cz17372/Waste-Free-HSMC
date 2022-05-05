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
        plot(ss*collect(1:P),out[1,:],xlabel=L"\tau",ylabel="Jump Variance",label="",color=c[11],framestyle=:box)
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
function plotTrajESS(R,M,P;col="Blues",fig=nothing)
    T = length(R.λ) -1
    c = colormap(col,T+10);
    out = TrajESS(R,M,P)
    if isnothing(fig)
        plot(out[1,:],xlabel="t",ylabel="%ESS",label="",color=c[1])
        for n=2:T
            plot!(out[n,:],label="",color=c[n])
        end
        current()
    else
        for n=1:T
            plot!(fig,out[n,:],label="",color=c[n])
        end
        current()
    end
end


N = 10000; M = 100; P = div(N,M)
include("src/WasteFree4.jl")
R = WasteFree.SMC(N,M,model=sonar,ϵ=0.05,α=0.5,method="full",mass_mat="identity",printl=true);
fig1 = plotJumpVar(R,0.1,M,P)
fig2 = plotTrajESS(R,M,P)
plotTrajESS(R,M,P,col="Purples",fig=fig2)
JV = JumpVariance(R,M,P)
plot(mapslices(getmaxidx,JV,dims=2)[:,1] * 0.1)

N = 30000; M = 100; P = div(N,M)
R = WasteFree.SMC(N,M,model=sonar,ϵ=20/P,α=0.2,method="full",mass_mat="identity",printl=true);
LogNC(R)

λ = R.λ
τ = mapslices(getmaxidx,JV,dims=2)[:,1]*0.1
include("src/WasteFree3.jl")
N = 10000; M = 100; P = div(N,M)
R = WasteFree.SMC(N,M,model=sonar,λ=λ,ϵ=τ/P,α=0.5,method="full",mass_mat="identity",printl=true);
LogNC(R)
using JLD2,SharedArrays
@save "data.jld2" λ τ
using DataFrames, CSV, StatsPlots, PlotlyJS
bp1 = box(y=data2.N30000M50_NC,name="M=50")
bp2 = box(y=data2.N30000M150_NC,name="M=150")
bp3 = box(y=data2.N30000M200_NC,name="M=200")
bp4 = box(y=data2.N30000M250_NC,name="M=250")
bp5 = box(y=data2.N30000M300_NC,name="M=300")
bp6 = box(y=data2.N30000M400_NC,name="M=400")
bp7 = box(y=data2.N30000M500_NC,name="M=500")
bp8 = box(y=data2.N30000M600_NC,name="M=600")
PlotlyJS.plot([bp1,bp2,bp3,bp4,bp5,bp6,bp7,bp8],Layout(yaxis_range=(-140,-110)))
data = CSV.read("data/sonar/full3.csv",DataFrame)
data2 = CSV.read("data/sonar/full4.csv",DataFrame)
bp4 = box(y=data.N10000M50eps5alpha50_NC,name="M=50,fix_eps=0.05")
bp5 = box(y=data.N10000M50eps15alpha50_NC,name="M=50,fix_eps=0.15")
bp6 = box(y=data.N10000M50eps30alpha50_NC,name="M=50,fix_eps=0.2")
bp7 = box(y=data.N10000M100eps5alpha50_NC,name="M=100,fix_eps=0.05")
bp8 = box(y=data.N10000M100eps15alpha50_NC,name="M=100,fix_eps=0.15")
bp9 = box(y=data.N10000M100eps30alpha50_NC,name="M=100,fix_eps=0.2")
bp10 = box(y=data.N10000M200eps5alpha50_NC,name="M=200,fix_eps=0.05")
bp11 = box(y=data.N10000M200eps15alpha50_NC,name="M=200,fix_eps=0.15")
bp12 = box(y=data.N10000M200eps25alpha50_NC,name="M=200,fix_eps=0.25")
bp13 = box(y=spec.NC,name="M=200,fix_eps=0.35")
spec = CSV.read("spec.csv",DataFrame)

PlotlyJS.plot([bp1,bp4,bp5,bp6],Layout(yaxis_range=(-140,-110)))
PlotlyJS.plot([bp2,bp7,bp8,bp9],Layout(yaxis_range=(-140,-110)))
PlotlyJS.plot([bp3,bp10,bp11,bp12,bp13],Layout(yaxis_range=(-140,-110)))

bp10 = box(y=data.N10000M200eps10alpha50_NC,name="M=200,fix_eps=0.1")
bp11 = box(y=data.N10000M200eps15alpha50_NC,name="M=200,fix_eps=0.15")
bp12 = box(y=data.N10000M200eps20alpha50_NC,name="M=200,fix_eps=0.2")

M20 = CSV.read("M20.csv",DataFrame)


    
