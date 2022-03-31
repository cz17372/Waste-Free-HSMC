using Distributed, SharedArrays, JLD2
NumWorkers = readline("Get Number of Workers")
NumWorkers = parse(Int64,NumWorkers)
addprocs(NumWorkers)
@everywhere include("sonar.jl")
NC = SharedArray{Float64}(100)
E  = SharedArray{Float64}(100)
N = readline("Input No. Particles..")
N = parse(Int64,N)
M = readline("Input the value of M..")
M = parse(Int64,M)
ϵ = readline("Input ϵ...")
ϵ = parse(Float64,ϵ)
α = readline("Input α...")
α = parse(Float64,α)
filename = readline("Input Data File Name")
@distributed for i = 1:100
    println("Running Simulation $(i)")
    R = SMC(N,M,U0,U,61,α,ϵ,initDist)
    NC[i] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E[i]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end


Info = "N = "*string(N)*" M = "*string(M)*" ϵ = "*string(ϵ)*" α = "*string(α)

@save filename Info NC E