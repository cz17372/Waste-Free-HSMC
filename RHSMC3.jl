using Distributed, SharedArrays,DataFrames,CSV
function run_exp(N,M,ϵ,α,λ,model,method;mass_mat,silence=true)
    Time = SharedArray{Float64}(100)
    NC   = SharedArray{Float64}(100)
    MM  = SharedArray{Float64}(100)
    @sync @distributed for n = 1:100
        if !silence
            println("Running Simulation for index $(n)")
        end
        t = @timed R = WasteFree.SMC(N,M,model=model,λ=λ,ϵ=ϵ*ones(length(λ)-1),α=α,method=method,mass_mat=mass_mat);
        Time[n] = t.time - t.gctime
        NC[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
    end
    filename = "data/"*model.name*"/"*method*"5.csv"
    df = CSV.read(filename,DataFrame)
    dat_name = "N$(N)M$(M)eps$(Int(ϵ*100))alpha$(Int(α*100))"
    df[!,dat_name*"_Time"] = Time
    df[!,dat_name*"_NC"] = NC
    df[!,dat_name*"_MM"] = MM
    CSV.write(filename,df)
    rmprocs(procs()[2:end])
    return nothing
end
println("Enter the number of workers")
nprocs = readline()
nprocs = parse(Int64,nprocs)
println("Enter N")
N = readline()
N = parse(Int64,N)
N = [N];
M = [100,75,125];
ϵ = [0.05,0.1,0.15,0.2,0.25,0.3,0.35]
α = [0.5]
df = DataFrame("exprid" => collect(1:100))
CSV.write("data/sonar/full5.csv",df)
for n in N
    for m in M
        for eps in ϵ
            for al in α
                addprocs(nprocs)
                @everywhere include("Models/sonar.jl")
                @everywhere include("src/WasteFree3.jl")
                @everywhere using Distributed, DistributedArrays,JLD2
                @everywhere using Statistics, StatsBase
                @everywhere @load "data.jld2" λ τ
                println("Running experiments for N = $(n), M = $(m), ϵ = $(eps), α=$(al), mass_mat = identity,method=full")
                run_exp(n,m,eps,al,λ,sonar,"full",mass_mat="identity",silence=false)
                rmprocs(procs()[2:end])
            end
        end
    end
end