using Distributed, SharedArrays, Measures, JLD2
addprocs(2)
@everywhere include("sonar.jl")
NC = SharedArray{Float64}(100)
E  = SharedArray{Float64}(100)
@distributed for n = 1:100
    println("Running Simulation $(n)")
    R = SMC(200000,1000,U0,U,61,0.5,0.2,initDist)
    NC[n] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E[n]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end

using Plots, StatsPlots
boxplot(NC,ylim = (-135,-115),framestyle=:box,label="",color=:grey,size=(300,500),ylabel="logLT",margin=20pt)
boxplot(E,ylim = (0.36,0.48),framestyle=:box,label="",color=:grey,size=(300,500),ylabel="mean of all marginals",margin=20pt)

Info = "N = 200000, ϵ=0.2, α = 0.5, M = 1000, NoTunning"
@save "Results_NoTuning_200000Pars_2.jld2" Info NC E

