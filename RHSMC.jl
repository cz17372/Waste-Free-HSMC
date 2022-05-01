using Distributed, SharedArrays,DataFrames,CSV
function run_exp(N,M,ϵ,α,model,method;mass_mat,silence=true)
    Time = SharedArray{Float64}(100)
    NC   = SharedArray{Float64}(100)
    MM  = SharedArray{Float64}(100)
    @sync @distributed for n = 1:100
        if !silence
            println("Running Simulation for index $(n)")
        end
        t = @timed R = WasteFree.SMC(N,M,model=model,ϵ=ϵ,α=α,method=method,mass_mat=mass_mat);
        Time[n] = t.time - t.gctime
        NC[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
    end
    filename = "data/"*model.name*"/"*method*".csv"
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
N = [10000];
M = [20,50,100,200,500];
ϵ = [0.1,0.2,0.3]
α = [0.5,0.7,0.9]
df = DataFrame("exprid" => collect(1:100))
CSV.write("data/sonar/full.csv",df)
for n in N
    for m in M
        for eps in ϵ
            for al in α
                addprocs(nprocs)
                @everywhere include("Models/sonar.jl")
                @everywhere include("src/WasteFree.jl")
                @everywhere using Distributed, DistributedArrays
                @everywhere using Statistics, StatsBase
                println("Running experiments for N = $(n), M = $(m), ϵ = $(eps), α=$(al), mass_mat = identity,method=full")
                run_exp(n,m,eps,al,sonar,"full",mass_mat="identity",silence=false)
                rmprocs(procs()[2:end])
            end
        end
    end
end