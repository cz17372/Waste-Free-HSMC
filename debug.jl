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
        """
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
        """
        t = @timed R = WasteFree.SMC(N,M,model=model,ϵ=ϵ,α=α,method="chopin",mass_mat=mass_mat);
        Time_Chopin[n] = t.time - t.gctime
        NC_Chopin[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM_Chopin[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
    end
    return DataFrame(Time_Sequential=Time_Sequential,Time_Independent=Time_Independent,Time_Full=Time_Full,Time_Chopin=Time_Chopin,NC_Sequential=NC_Sequential,NC_Independent=NC_Independent,NC_Full=NC_Full,NC_Chopin=NC_Chopin,MM_Sequential=MM_Sequential,MM_Independent=MM_Independent,MM_Full=MM_Full,MM_Chopin=MM_Chopin)
end
