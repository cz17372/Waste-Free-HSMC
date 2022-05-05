using Distributed, SharedArrays
addprocs(5)
@everywhere using JLD2, StatsBase
@everywhere include("Models/sonar.jl")
@everywhere include("src/WasteFree3.jl")
@everywhere @load "data.jld2"
NC   = SharedArray{Float64}(100)
MM  = SharedArray{Float64}(100)
println("Enter N")
N = readline()
N = parse(Int64,N)
println("Enter M")
M = readline()
M = parse(Int64,M)
P = div(N,M)
@sync @distributed for n = 1:100
    println("Running Simulation for index $(n)")
    R = WasteFree.SMC(N,M,model=sonar,λ=λ,ϵ=0.35*ones(length(λ)-1),α=0.5,method="full",mass_mat="identity");
    NC[n] = sum(log.(mean(exp.(R.logW),dims=1)))
    MM[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
end

