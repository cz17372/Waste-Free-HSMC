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

N = 40000; M = 100; P = div(N,M)
include("src/WasteFree4.jl")
R = WasteFree.SMC(N,M,model=sonar,ϵ=0.1,α=0.5,method="full",mass_mat="identity",printl=true);
fig1 = plotJumpVar(R,0.1,M,P)
JV = JumpVariance(R,M,P)
plot(mapslices(getmaxidx,JV,dims=2)[:,1] * 0.1)

N = 30000; M = 100; P = div(N,M)
R = WasteFree.SMC(N,M,model=sonar,ϵ=20/P,α=0.5,method="full",mass_mat="identity",printl=true);
LogNC(R)

λ = R.λ
τ = mapslices(getmaxidx,JV,dims=2)[:,1]*0.1
include("src/WasteFree3.jl")
N = 10000; M = 100; P = div(N,M)
R = WasteFree.SMC(N,M,model=sonar,λ=λ,ϵ=τ/P,α=0.5,method="full",mass_mat="identity",printl=true);
LogNC(R)
using JLD2,SharedArrays
@save "data.jld2" λ τ
@load "M100.jld2" NC MM

using DataFrames, CSV, StatsPlots, PlotlyJS
M200 = CSV.read("M200.csv",DataFrame)
M50 = CSV.read("M50.csv",DataFrame)
M50 = CSV.read("M50.csv",DataFrame)
bp1 = box(y=M50.NC,name="M=50")
bp2 = box(y=M100.NC,name="M=100")
bp3 = box(y=M200.NC,name="M=200")
bp0 = box(y=M20.NC,name="M=20")
PlotlyJS.plot([bp0,bp1,bp2,bp3],Layout(yaxis_range=(-140,-110)))
data = CSV.read("data/sonar/full.csv",DataFrame)
bp4 = box(y=data.N10000M50eps10alpha50_NC,name="M=50,fix_eps=0.1")
bp5 = box(y=data.N10000M50eps15alpha50_NC,name="M=50,fix_eps=0.15")
bp6 = box(y=data.N10000M50eps20alpha50_NC,name="M=50,fix_eps=0.2")
bp7 = box(y=data.N10000M100eps10alpha50_NC,name="M=100,fix_eps=0.1")
bp8 = box(y=data.N10000M100eps15alpha50_NC,name="M=100,fix_eps=0.15")
bp9 = box(y=data.N10000M100eps20alpha50_NC,name="M=100,fix_eps=0.2")
bp10 = box(y=data.N10000M200eps10alpha50_NC,name="M=200,fix_eps=0.1")
bp11 = box(y=data.N10000M200eps15alpha50_NC,name="M=200,fix_eps=0.15")
bp12 = box(y=data.N10000M200eps20alpha50_NC,name="M=200,fix_eps=0.2")


PlotlyJS.plot([bp1,bp4,bp5,bp6],Layout(yaxis_range=(-140,-110)))
PlotlyJS.plot([bp2,bp7,bp8,bp9],Layout(yaxis_range=(-140,-110)))
PlotlyJS.plot([bp3,bp10,bp11,bp12],Layout(yaxis_range=(-140,-110)))

bp10 = box(y=data.N10000M200eps10alpha50_NC,name="M=200,fix_eps=0.1")
bp11 = box(y=data.N10000M200eps15alpha50_NC,name="M=200,fix_eps=0.15")
bp12 = box(y=data.N10000M200eps20alpha50_NC,name="M=200,fix_eps=0.2")

M20 = CSV.read("M20.csv",DataFrame)