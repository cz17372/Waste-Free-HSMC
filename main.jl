using Distributed, SharedArrays, Measures, JLD2
addprocs(4)
@everywhere include("sonar.jl")
NC = SharedArray{Float64}(100)
E  = SharedArray{Float64}(100)
@distributed for n = 1:100
    println("Running Simulation $(n)")
<<<<<<< HEAD
    R = SMC(10000,100,U0,U,61,0.5,0.2,initDist,1)
=======
    R = SMC(10000,100,U0,U,61,0.5,0.5,initDist)
>>>>>>> parent of 6e48659 (....)
    NC[n] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E[n]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end
