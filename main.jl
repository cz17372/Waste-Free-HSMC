using Distributed, SharedArrays, Measures, JLD2, Plots, StatsPlots, DataFrames
addprocs(4)
@everywhere using StatsBase, Statistics
@everywhere include("src/WasteFree.jl")
@everywhere include("Models/sonar.jl")
NC = SharedArray{Float64}(100)
E  = SharedArray{Float64}(100)
Time = SharedArray{Float64}(100)
@distributed for i = 1:100
    println("Start Simulation $(i)")
    t = @timed R =  WasteFree.SMC(50000,100,model=sonar,ϵ=0.2,α=0.5,method="full");
    Time[i] = t.time
    NC[i] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E[i]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end
NC2 = SharedArray{Float64}(100)
E2  = SharedArray{Float64}(100)
Time2 = SharedArray{Float64}(100)
@distributed for i = 1:100
    println("Start Simulation $(i)")
    t = @timed R =  WasteFree.SMC(50000,100,model=sonar,ϵ=0.2,α=0.5,method="chopin");
    Time2[i] = t.time
    NC2[i] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E2[i]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end
P1 = boxplot(NC,label="Hamiltonian",size=(400,800),margin=15pt,ylim=(-140,-110),framestyle=:box,color=:grey,ylabel="log Normalising Constants");
P2 = boxplot(NC2,label="Metropolis-Hastings",size=(400,800),margin=15pt,ylim=(-140,-110),framestyle=:box,color=:darkolivegreen);
plot(P1,P2,layout=(1,2),size=(800,800))
P1 = boxplot(E,label="Hamiltonian",size=(400,800),margin=15pt,ylim=(0.36,0.48),framestyle=:box,color=:grey,ylabel="mean of marginals");
P2 = boxplot(E2,label="Metropolis-Hastings",size=(400,800),margin=15pt,ylim=(0.36,0.48),framestyle=:box,color=:darkolivegreen);
plot(P1,P2,layout=(1,2),size=(800,800))
P1 = boxplot(Time,label="Hamiltonian",size=(400,800),margin=15pt,framestyle=:box,color=:grey,ylabel="Execution time(s)");
P2 = boxplot(Time2,label="Metropolis-Hastings",size=(400,800),margin=15pt,framestyle=:box,color=:darkolivegreen);
plot(P1,P2,layout=(1,2),size=(800,800))
