using Distributed, SharedArrays, Measures, JLD2, Plots, StatsPlots
addprocs(4)
@everywhere include("sonar.jl")
NC = SharedArray{Float64}(100)
E  = SharedArray{Float64}(100)
@distributed for i = 1:100
    println("Start Simulation $(i)")
    R =  SMC(10000,50,U0,U,61,0.5,0.2,initDist,2)
    NC[i] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E[i]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end

P1 = boxplot(NC,label="",ylim=(-135,-115))
P2 = boxplot(NC2,label="",ylim=(-135,-115))
NC2 = SharedArray{Float64}(100)
E2  = SharedArray{Float64}(100)
@distributed for i = 1:100
    println("Start Simulation $(i)")
    R =  SMC(10000,100,U0,U,61,0.5,0.2,initDist,1)
    NC2[i] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E2[i]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end

plot(P1,P2,layout=(1,2))