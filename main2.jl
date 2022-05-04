using Distributions,LinearAlgebra, StatsBase
include("Models/banana.jl")
include("Models/sonar.jl")
include("src/HMC.jl")
include("src/WasteFree2.jl")
R,acc = HMC.run(100000,1.0,20,U=banana.U,x0=randn(2),acc_prob=true)
lines(R[:,2])

@time R = WasteFree.SMC(10000,100,model=sonar,ϵ=0.2,α=0.5,method="full",mass_mat="identity",printl=true);

function LogNC(R)
    return sum(log.(mean(exp.(R.logW),dims=1)[1,:]))
end

LogNC(R)

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

using Optim, Roots
f_univariate(x) = 2x^2+3x+1
@time find_zeros(f_univariate,-10,10);
@time find_zero(f_univariate,(-0.6,1.0));

Optim.minimum(res)

res.minimum