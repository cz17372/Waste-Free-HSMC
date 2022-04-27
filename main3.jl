using Distributed, SharedArrays,DataFrames,CSV
println("Enter the number of workers...")
NWorker = readline()
NWorker = parse(Int64,NWorker)
addprocs(NWorker)
@everywhere include("Models/sonar.jl")
@everywhere include("src/WasteFree.jl")
@everywhere using Distributed, DistributedArrays
@everywhere using Statistics, StatsBase
function run_exp(N,M,ϵ,α,model,method;mass_mat,silence=true)
    Time = SharedArray{Float64}(100)
    NC   = SharedArray{Float64}(100)
    NC  = SharedArray{Float64}(100)
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
    return DataFrame("exprid"=>collect(1:100),"Time_"*method=>Time,"NC_"*method=>NC,"MM_"*method=>MM)
end

function overallexp(N,M,ϵ,α,model;mass_mat,silence=true)
    R  = DataFrame("exprid"=>collect(1:100))
    println("Running WF-Sequential...")
    R_Sequential = run_exp(N,M,ϵ,α,model,"sequential",mass_mat=mass_mat,silence=silence)
    R = leftjoin(R,R_Sequential,on="exprid")
    println("Running WF-Independent...")
    R_Independent = run_exp(N,M,ϵ,α,model,"independent",mass_mat=mass_mat,silence=silence)
    R = leftjoin(R,R_Independent,on="exprid")
    println("Running WF-Full...")
    R_Full = run_exp(N,M,ϵ,α,model,"full",mass_mat=mass_mat,silence=silence)
    R = leftjoin(R,R_Full,on="exprid")
    println("Running WF-Chopin...")
    R_Chopin = run_exp(N,M,ϵ,α,model,"chopin",mass_mat=mass_mat,silence=silence)
    R = leftjoin(R,R_Chopin,on="exprid")
    return R
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

filename = filename*"alpha50indentity.csv"
println("Running experiments for N = $(N), M = $(M), ϵ = $(ϵ)")
R1 = overallexp(N,M,ϵ,0.5,sonar,mass_mat="identity",silence=false)
CSV.write(filename,R1)
