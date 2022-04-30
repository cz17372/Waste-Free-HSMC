using Distributions, CairoMakie, StatsPlots, LinearAlgebra
include("Models/banana.jl")
include("src/HMC.jl")
R,acc = HMC.run(100000,1.0,20,U=banana.U,x0=randn(2),acc_prob=true)
lines(R[:,2])

CairoMakie.scatter(R[:,1],R[:,2])

x = range(-40,40,step=0.1)
y = range(-20,20,step=0.1)
z = zeros(length(x),length(y))
for i = 1:length(x)
    for j = 1:length(y)
        z[i,j] = banana.U([x[i],y[j]],requires_grad=false)
    end
end

CairoMakie.heatmap(x,y,z,colorrange=(0,10),colormap=Reverse(:aquamarine))
CairoMakie.scatter!(R[end-200:end,1],R[end-200:end,2],markersize=5)
current_figure()
