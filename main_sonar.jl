using Distributed, SharedArrays, JLD2
println("Get Number of Workers")
NumWorkers = readline()
NumWorkers = parse(Int64,NumWorkers)
addprocs(NumWorkers)
@everywhere include("sonar.jl")
NC = SharedArray{Float64}(100)
E  = SharedArray{Float64}(100)
println("Input N...")
N = readline()
N = parse(Int64,N)
println("Input the value of M..")
M = readline()
M = parse(Int64,M)
println("Input ϵ...")
ϵ = readline()
ϵ = parse(Float64,ϵ)
println("Input α...")
α = readline()
α = parse(Float64,α)
println("Input Data File Name")
filename = readline()
@distributed for i = 1:100
    println("Running Simulation $(i)")
    R = SMC(N,M,U0,U,61,α,ϵ,initDist)
    NC[i] = sum(log.(mean(exp.(R.logW[:,2:end]),dims=1)))
    E[i]  = sum(R.W[:,end].*mean(R.X[end],dims=2))
end


Info = "N = "*string(N)*" M = "*string(M)*" ϵ = "*string(ϵ)*" α = "*string(α)

@save filename Info NC E