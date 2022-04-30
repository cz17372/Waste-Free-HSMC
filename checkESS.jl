using Distributions, Plots, StatsPlots
include("src/WasteFree.jl")
include("Models/sonar.jl")
function LogNC(R)
    return sum(log.(mean(exp.(R.logW),dims=1)[1,:]))
end
lambda_chopin = Array{Any,1}(undef,20)
for n = 1:20
    R = WasteFree.SMC(10000,100,model=sonar,ϵ=0.2,α=0.5,method="chopin",mass_mat="identity",printl=true);
    lambda_chopin[n] = R.λ
end
theme(:ggplot2)
plot(lambda_chopin[1],label="",color=:darkolivegreen)
for n=2:20
    plot!(lambda_full[n],label="",color=:darkolivegreen)
end
current()

lambda_full = Array{Any,1}(undef,20)
for n = 1:20
    R = WasteFree.SMC(10000,100,model=sonar,ϵ=0.2,α=0.5,method="full",mass_mat="identity",printl=true);
    lambda_chopin[n] = R.λ
end

for n=1:20
    plot!(lambda_chopin[n],label="",color=:darkolivegreen)
end
current()
