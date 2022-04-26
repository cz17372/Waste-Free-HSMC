using Distributed, SharedArrays,DataFrames,CSV
println("Enter the number of workers...")
NWorker = readline()
NWorker = parse(Int64,NWorker)
addprocs(NWorker)
@everywhere include("Models/sonar.jl")
@everywhere include("src/WasteFree.jl")
@everywhere using Statistics, StatsBase
function run_exp(N,M,ϵ,α,model,mass_mat;silence=true)
    Time_Sequential = SharedArray{Float64}(100)
    Time_Independent= SharedArray{Float64}(100)
    Time_Full       = SharedArray{Float64}(100)
    Time_Chopin     = SharedArray{Float64}(100)
    NC_Sequential   = SharedArray{Float64}(100)
    NC_Independent  = SharedArray{Float64}(100)
    NC_Full         = SharedArray{Float64}(100)
    NC_Chopin       = SharedArray{Float64}(100)
    MM_Sequential   = SharedArray{Float64}(100)
    MM_Independent  = SharedArray{Float64}(100) 
    MM_Full         = SharedArray{Float64}(100)
    MM_Chopin       = SharedArray{Float64}(100)
    @sync @distributed for n = 1:100
        if !silence
            println("Running WFSMC-Sequential for index $(n)")
        end
        t = @timed R = WasteFree.SMC(N,M,model=model,ϵ=ϵ,α=α,method="sequential",mass_mat=mass_mat);
        Time_Sequential[n] = t.time - t.gctime
        NC_Sequential[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM_Sequential[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
        if !silence
            println("Running WFSMC-Independent for index $(n)")
        end
        t = @timed R = WasteFree.SMC(N,M,model=model,ϵ=ϵ,α=α,method="independent",mass_mat=mass_mat);
        Time_Independent[n] = t.time - t.gctime
        NC_Independent[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM_Independent[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
        if !silence
            println("Running WFSMC-Full for index $(n)")
        end
        t = @timed R = WasteFree.SMC(N,M,model=model,ϵ=ϵ,α=α,method="full");
        Time_Full[n] = t.time
        NC_Full[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM_Full[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
        if !silence
            println("Running WFSMC-Chopin for index $(n)")
        end
        t = @timed R = WasteFree.SMC(N,M,model=model,ϵ=ϵ,α=α,method="chopin",mass_mat=mass_mat);
        Time_Chopin[n] = t.time - t.gctime
        NC_Chopin[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM_Chopin[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
    end
    return DataFrame(Time_Sequential=Time_Sequential,Time_Independent=Time_Independent,Time_Full=Time_Full,Time_Chopin=Time_Chopin,NC_Sequential=NC_Sequential,NC_Independent=NC_Independent,NC_Full=NC_Full,NC_Chopin=NC_Chopin,MM_Sequential=MM_Sequential,MM_Independent=MM_Independent,MM_Full=MM_Full,MM_Chopin=MM_Chopin)
end


# Performing Experiments For Sonar Example
# N = 10000, M = 50,100,200, α = 0.5
# N = 50000, M = 50,100,200, α = 0.5
# N = 100000,M = 50,100,200,800, α = 0.5

println("Runing experiment with N = 10000, M = 50,α = 0.5, ϵ=0.1")
R = run_exp(10000,50,0.1,0.5,sonar,"identity")
CSV.write("N1e4M50eps01alpha05.csv",R)
println("Runing experiment with N = 10000, M = 50,α = 0.5, ϵ=0.2")
R = run_exp(10000,50,0.2,0.5,sonar,"identity")
CSV.write("N1e4M50eps02alpha05.csv",R)
println("Runing experiment with N = 10000, M = 50,α = 0.5, ϵ=0.3")
R = run_exp(10000,50,0.3,0.5,sonar,"identity")
CSV.write("N1e4M50eps03alpha05.csv",R)

println("Runing experiment with N = 10000, M = 100,α = 0.5, ϵ=0.1")
R = run_exp(10000,100,0.1,0.5,sonar,"identity")
CSV.write("N1e4M100eps01alpha05.csv",R)
println("Runing experiment with N = 10000, M = 100,α = 0.5, ϵ=0.2")
R = run_exp(10000,100,0.2,0.5,sonar,"identity")
CSV.write("N1e4M100eps02alpha05.csv",R)
println("Runing experiment with N = 10000, M = 100,α = 0.5, ϵ=0.3")
R = run_exp(10000,100,0.3,0.5,sonar,"identity")
CSV.write("N1e4M100eps03alpha05.csv",R)


println("Runing experiment with N = 10000, M = 200,α = 0.5, ϵ=0.1")
R = run_exp(10000,200,0.1,0.5,sonar,"identity")
CSV.write("N1e4M200eps01alpha05.csv",R)
println("Runing experiment with N = 10000, M = 200,α = 0.5, ϵ=0.2")
R = run_exp(10000,200,0.2,0.5,sonar,"identity")
CSV.write("N1e4M200eps02alpha05.csv",R)
println("Runing experiment with N = 10000, M = 200,α = 0.5, ϵ=0.3")
R = run_exp(10000,200,0.3,0.5,sonar,"identity")
CSV.write("N1e4M200eps03alpha05.csv",R)

#-------------------------------------------

println("Runing experiment with N = 50000, M = 50,α = 0.5, ϵ=0.1")
R = run_exp(50000,50,0.1,0.5,sonar,"identity")
CSV.write("N5e4M50eps01alpha05.csv",R)
println("Runing experiment with N = 50000, M = 50,α = 0.5, ϵ=0.2")
R = run_exp(50000,50,0.2,0.5,sonar,"identity")
CSV.write("N5e4M50eps02alpha05.csv",R)
println("Runing experiment with N = 50000, M = 50,α = 0.5, ϵ=0.3")
R = run_exp(50000,50,0.3,0.5,sonar,"identity")
CSV.write("N5e4M50eps03alpha05.csv",R)

println("Runing experiment with N = 50000, M = 100,α = 0.5, ϵ=0.1")
R = run_exp(50000,100,0.1,0.5,sonar,"identity")
CSV.write("N5e4M100eps01alpha05.csv",R)
println("Runing experiment with N = 50000, M = 100,α = 0.5, ϵ=0.2")
R = run_exp(50000,100,0.2,0.5,sonar,"identity")
CSV.write("N5e4M100eps02alpha05.csv",R)
println("Runing experiment with N = 50000, M = 100,α = 0.5, ϵ=0.3")
R = run_exp(50000,100,0.3,0.5,sonar,"identity")
CSV.write("N5e4M100eps03alpha05.csv",R)


println("Runing experiment with N = 50000, M = 200,α = 0.5, ϵ=0.1")
R = run_exp(50000,200,0.1,0.5,sonar,"identity")
CSV.write("N5e4M200eps01alpha05.csv",R)
println("Runing experiment with N = 50000, M = 200,α = 0.5, ϵ=0.2")
R = run_exp(50000,200,0.2,0.5,sonar,"identity")
CSV.write("N5e4M200eps02alpha05.csv",R)
println("Runing experiment with N = 50000, M = 200,α = 0.5, ϵ=0.3")
R = run_exp(50000,200,0.3,0.5,sonar,"identity")
CSV.write("N5e4M200eps03alpha05.csv",R)


#-------------------------------------------

println("Runing experiment with N = 100000, M = 50,α = 0.5, ϵ=0.1")
R = run_exp(100000,50,0.1,0.5,sonar,"identity")
CSV.write("N1e5M50eps01alpha05.csv",R)
println("Runing experiment with N = 100000, M = 50,α = 0.5, ϵ=0.2")
R = run_exp(100000,50,0.2,0.5,sonar,"identity")
CSV.write("N1e5M50eps02alpha05.csv",R)
println("Runing experiment with N = 100000, M = 50,α = 0.5, ϵ=0.3")
R = run_exp(100000,50,0.3,0.5,sonar,"identity")
CSV.write("N1e5M50eps03alpha05.csv",R)

println("Runing experiment with N = 100000, M = 100,α = 0.5, ϵ=0.1")
R = run_exp(100000,100,0.1,0.5,sonar,"identity")
CSV.write("N1e5M100eps01alpha05.csv",R)
println("Runing experiment with N = 100000, M = 100,α = 0.5, ϵ=0.2")
R = run_exp(100000,100,0.2,0.5,sonar,"identity")
CSV.write("N1e5M100eps02alpha05.csv",R)
println("Runing experiment with N = 100000, M = 100,α = 0.5, ϵ=0.3")
R = run_exp(100000,100,0.3,0.5,sonar,"identity")
CSV.write("N1e5M100eps03alpha05.csv",R)

println("Runing experiment with N = 100000, M = 200,α = 0.5, ϵ=0.1")
R = run_exp(100000,200,0.1,0.5,sonar,"identity")
CSV.write("N1e5M200eps01alpha05.csv",R)
println("Runing experiment with N = 100000, M = 200,α = 0.5, ϵ=0.2")
R = run_exp(100000,200,0.2,0.5,sonar,"identity")
CSV.write("N1e5M200eps02alpha05.csv",R)
println("Runing experiment with N = 100000, M = 200,α = 0.5, ϵ=0.3")
R = run_exp(100000,200,0.3,0.5,sonar,"identity")
CSV.write("N1e5M200eps03alpha05.csv",R)
#-------------------------------------------
println("Runing experiment with N = 200000, M = 50,α = 0.5, ϵ=0.1")
R = run_exp(200000,50,0.1,0.5,sonar,"identity")
CSV.write("N2e5M50eps01alpha05.csv",R)
println("Runing experiment with N = 200000, M = 50,α = 0.5, ϵ=0.2")
R = run_exp(200000,50,0.2,0.5,sonar,"identity")
CSV.write("N2e5M50eps02alpha05.csv",R)
println("Runing experiment with N = 200000, M = 50,α = 0.5, ϵ=0.3")
R = run_exp(200000,50,0.3,0.5,sonar,"identity")
CSV.write("N2e5M50eps03alpha05.csv",R)

println("Runing experiment with N = 200000, M = 100,α = 0.5, ϵ=0.1")
R = run_exp(200000,100,0.1,0.5,sonar,"identity")
CSV.write("N2e5M100eps01alpha05.csv",R)
println("Runing experiment with N = 200000, M = 100,α = 0.5, ϵ=0.2")
R = run_exp(200000,100,0.2,0.5,sonar,"identity")
CSV.write("N2e5M100eps02alpha05.csv",R)
println("Runing experiment with N = 200000, M = 100,α = 0.5, ϵ=0.3")
R = run_exp(200000,100,0.3,0.5,sonar,"identity")
CSV.write("N2e5M100eps03alpha05.csv",R)


println("Runing experiment with N = 200000, M = 200,α = 0.5, ϵ=0.1")
R = run_exp(200000,200,0.1,0.5,sonar,"identity")
CSV.write("N2e5M200eps01alpha05.csv",R)
println("Runing experiment with N = 200000, M = 200,α = 0.5, ϵ=0.2")
R = run_exp(200000,200,0.2,0.5,sonar,"identity")
CSV.write("N2e5M200eps02alpha05.csv",R)
println("Runing experiment with N = 200000, M = 200,α = 0.5, ϵ=0.3")
R = run_exp(200000,200,0.3,0.5,sonar,"identity")
CSV.write("N2e5M200eps03alpha05.csv",R)

