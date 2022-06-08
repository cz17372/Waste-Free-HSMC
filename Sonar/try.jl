include("WR.jl")
include("sonar.jl")
wasterecycling.SMC(10000,100,model=sonar,ϵ=0.2,α=0.5,mass_mat="identity",printl=true)