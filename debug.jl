using StatsBase, Statistics
include("src/WasteFree.jl")
include("Models/sonar.jl")

@time R = WasteFree.SMC(50000,100,model=sonar,ϵ=0.1,α=0.5,method="full",printl=true);
sum(log.(mean(exp.(R.logW),dims=1)))