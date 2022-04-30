using Distributions, CairoMakie, StatsPlots
include("src/WasteFree.jl")
include("Models/sonar.jl")
function LogNC(R)
    return sum(log.(mean(exp.(R.logW),dims=1)[1,:]))
end
R = WasteFree.SMC(50000,40,model=sonar,ϵ=0.2,α=0.5,method="full",mass_mat="identity",printl=true);

M = 40; P = 50000÷M
function ESS(logW,M,P)
    N = M*P
    ess = zeros(P)
    for n = 1:P
        subw = exp.(logW[n:P:N])
        subw = subw/sum(subw)
        ess[n] = 1/sum(subw.^2)
    end
    return ess
end
LogNC(R)
lines(ESS(R.logW[:,11],M,P)/M)
f = Figure()
ax = Axis(f[1, 1], xlabel = "Trajectory Position", ylabel = "ESS/M")
lineobject = lines!(ax, 1:P, ESS(R.W[:,10],M,P)/M, color = :red)
f


function ESS2(logW,M,P)
    N = M*P
    ess = zeros(P)
    subw = zeros(M)
    sp = collect(1:P:M)
    for n = 1:P
        for i = 1:M
            subw[i] = sum(exp.(logW[sp[i]:sp[i]+n-1]))/n
        end
        subw = subw/sum(subw)
        ess[n] = 1/sum(subw.^2)
    end
    return ess
end

lines(ESS(R.logW[:,15],M,P)/M)
