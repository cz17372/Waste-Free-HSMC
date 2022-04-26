include("debug2.jl")
include("Models/sonar.jl")

t = @timed R = WasteFree.SMC(50000,100,model=sonar,ϵ=0.2,α=0.5,method="full",printl=true,mass_mat="identity");
using StatsBase
sum(log.(mean(exp.(R.logW),dims=1)))
sum(R.W[:,end].*mean(R.X[end],dims=2))