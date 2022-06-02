using Distributed, SharedArrays
addprocs(5)
@everywhere using JLD2
@everywhere include("debug.jl")
@everywhere @load "Orthant/L.jld2"
NC = SharedArray{Float64}(100)
MM = SharedArray{Float64}(100)
@sync @distributed for n = 1:100
    a = 1.5*ones(150); b= Inf*ones(150)
    println("Running Simulation for index $(n)")
    R = SMC(10000,100,0.2,L,a,b,150)
    NC[n] = R[4]
    MM[n] = sum(R[3][:,end].*mean(R[1][end],dims=1)[1,:])
    GC.gc()
end


using JLD2,Measures
include("SMC.jl"); @load "Orthant/L.jld2"
NC = zeros(100); MM = zeros(100);
for n = 1:100
    a = 1.5*ones(150); b= Inf*ones(150)
    println("Running Simulation for index $(n)")
    R = SMC(50000,500,0.2,L,a,b,150)
    NC[n] = R[4]
    MM[n] = sum(R[3][:,end].*(mean(R[1],dims=1)[1,:]))
end

