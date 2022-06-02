using Distributed, SharedArrays,DataFrames,CSV
function run_exp(N,M,ϵ,α,λ,model,method;mass_mat,filename,silence=true)
    Time = SharedArray{Float64}(100)
    NC   = SharedArray{Float64}(100)
    MM  = SharedArray{Float64}(100)
    @sync @distributed for n = 1:100
        if !silence
            println("Running Simulation for index $(n)")
        end
        t = @timed R = WasteFree.SMC(N,M,model=model,λ=λ,ϵ=ϵ,α=α,method=method,mass_mat=mass_mat);
        Time[n] = t.time - t.gctime
        NC[n] = sum(log.(mean(exp.(R.logW),dims=1)))
        MM[n] = sum(R.W[:,end].*mean(R.X[end],dims=2))
    end
    filename = "data/"*model.name*"/"*method*"4.csv"
    df = CSV.read(filename,DataFrame)
    dat_name = "N$(N)M$(M)"
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
N = [30000];
M = [50,75,100,125,150,200,250,300,400,500,600,1000,1500,2000];
df = DataFrame("exprid" => collect(1:100))
println("Enter the file name...")
filename = readline()
CSV.write("data/sonar/"*filename,df)
for n in N
    for m in M
        addprocs(nprocs)
        @everywhere include("Models/sonar.jl")
        @everywhere include("src/WasteFree3.jl")
        @everywhere using Distributed, DistributedArrays,JLD2
        @everywhere using Statistics, StatsBase
        @everywhere @load "data.jld2" λ τ
        println("Running experiments for N = $(n), M = $(m), mass_mat = identity,method=full")
        run_exp(n,m,τ/div(n,m),0.5,λ,sonar,"full",mass_mat="identity",silence=false)
        rmprocs(procs()[2:end])
    end
end