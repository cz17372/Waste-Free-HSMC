using Distributed, SharedArrays, Measures, JLD2
addprocs(4)
@everywhere include("sonar.jl")
NC = SharedArray{Float64}(100)
E  = SharedArray{Float64}(100)
