include("debug2.jl")
include("Models/sonar.jl")

t = @timed R = WasteFree.SMC(10000,50,model=sonar,ϵ=0.1,α=0.5,method="independent",printl=true,mass_mat="adaptive");

sum(log.(mean(exp.(R.logW),dims=1)))
sum(R.W[:,end].*mean(R.X[end],dims=2))