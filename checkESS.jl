using Distributions, CairoMakie, StatsPlots
include("src/WasteFree.jl")
include("Models/sonar.jl")
function LogNC(R)
    return sum(log.(mean(exp.(R.logW),dims=1)[1,:]))
end
R = WasteFree.SMC(10000,200,model=sonar,ϵ=0.2,α=0.8,method="full",mass_mat="identity",printl=true);

M = 50; P = 10000÷M
function ESS(logW,M,P)
    N = M*P
    ess = zeros(P)
    for n = 1:P
        subw = exp.(logW[n:P:N])
        subw = subw/sum(subw)
        ess[n] = 1/sum(subw.^2)
    end
    return ess
end
LogNC(R)
T = 22
f = Figure()
ax = Axis(f[1, 1], xlabel = "Trajectory Position", ylabel = "ESS/M",title="lambda = $(R.λ[T])")
lineobject = lines!(ax, 1:P, ESS(R.logW[:,T],M,P)/M, color = :red)
CairoMakie.ylims!(ax,(0.0,1.0))
f