using JLD2,Measures
include("SMC.jl"); @load "Orthant/L.jld2"
NC = zeros(100); MM = zeros(100);R = Array{Any,1}(undef,100);
Threads.@threads for n = 1:100
    a = 1.5*ones(150); b= Inf*ones(150); 
    println("Running Simulation for index $(n)")
    R[n] = SMC(10000,100,0.2,L,a,b,150)
    NC[n] = R[n][4]
    MM[n] = sum(R[n][3][:,end].*(mean(R[n][1],dims=1)[1,:]))
    println("LogNC=",NC[n],"MM=",MM[n])
end

using StatsPlots

@save "data/data1.jld2" NC MM

boxplot(NC,ylims=(-824,-815),yticks=-824:-815)