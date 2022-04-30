using Distributed, SharedArrays,DataFrames,CSV
println("Enter the number of workers...")
NWorker = readline()
NWorker = parse(Int64,NWorker)
addprocs(5)
@everywhere include("Models/sonar.jl")
@everywhere include("src/WasteFree.jl")
@everywhere using Distributed, DistributedArrays
@everywhere using Statistics, StatsBase
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
    dat_name = "N$(N)M$(M)alpha$(Int(α*100))"
    df[!,dat_name*"_Time"] = Time
    df[!,dat_name*"_NC"] = NC
    df[!,dat_name*"_MM"] = MM
    CSV.write(filename,df)
    return nothing
end
println("Enter the Number of Particles")
N = readline()
filename = "N"*N
N = parse(Int64,N)
println("Enter M")
M = readline()
filename = filename*"M"*M
M = parse(Int64,M)
println("Enter ϵ")
ϵ = readline()
filename = filename*"eps"*ϵ
ϵ = parse(Float64,ϵ)/100
println("Enter α")
α = readline()
filename = filename * α
α = parse(Float64,α)/100
println("Enter the mass matrix type")
mass_mat = readline()
filename = filename*mass_mat
println("Enter the method..")
method = readline()
filename = filename*method*".csv"
println("Running experiments for N = $(N), M = $(M), ϵ = $(ϵ), α=$(α), mass_mat = $(mass_mat),method=$(method)")
run_exp(N,M,ϵ,α,sonar,method,mass_mat=mass_mat,silence=false)