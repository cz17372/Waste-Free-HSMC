using Distributed, SharedArrays,DataFrames,CSV
function run_exp(N,M,α,model;silence=true)
    Time = SharedArray{Float64}(100)
    NC   = SharedArray{Float64}(100)
    MM  = SharedArray{Float64}(100)
    @sync @distributed for n = 1:100
        if !silence
            println("Running Simulation for index $(n)")
        end
        t=@timed _,_,_,_,NC[n],MM[n] = chopin.SMC(N,M,model=model,α=α);
        Time[n] = t.time
    end
    return NC,MM,Time
end


println("Enter N"); N = readline();println("Enter alpha"); alpha = readline();
println("Enter the address to store the results");file_addr = readline()
filename = "ChopinN"*N*"alpha"*alpha*"withtime.csv"
N = parse(Int64,N);alpha = parse(Float64,alpha)/100
traj_length = [100]
M = div.(Ref(N),traj_length)
df = DataFrame(ind=1:100)


for m in M
    addprocs(4)
    @everywhere include("sonar.jl")
    @everywhere include("chopin.jl")
    @everywhere using Distributed, DistributedArrays
    @everywhere using Statistics, StatsBase
    println("Running experiments for M = $(m), ϵ = $(eps)...")
    NC,MM,Time = run_exp(N,m,alpha,sonar,silence=false)
    df[!,"M"*string(m)*"_NC"] = NC
    df[!,"M"*string(m)*"_MM"] = MM
    df[!,"M"*string(m)*"_Time"] = Time
    rmprocs(procs()[2:end])
end

CSV.write(file_addr*filename,df)