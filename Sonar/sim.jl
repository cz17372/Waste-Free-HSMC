using Distributed, SharedArrays,DataFrames,CSV
function run_exp(N,M,ϵ,α,model;mass_mat,silence=true)
    Time = SharedArray{Float64}(100)
    NC   = SharedArray{Float64}(100)
    MM  = SharedArray{Float64}(100)
    @sync @distributed for n = 1:100
        if !silence
            println("Running Simulation for index $(n)")
        end
        t = @timed _,_,_,_,_,NC[n],MM[n] = wasterecycling.SMC(N,M,model=model,ϵ=ϵ,α=α,mass_mat=mass_mat);
        Time[n] = t.time
    end
    return NC,MM,Time
end

println("Enter the number of workers")
nprocs = readline()
nprocs = parse(Int64,nprocs)

println("Enter N"); N = readline();println("Enter alpha"); alpha = readline();
println("Enter the address to store the results");file_addr = readline()
filename = "N"*N*"alpha"*alpha*"withtime.csv"
N = parse(Int64,N);alpha = parse(Float64,alpha)/100
traj_length = [100]
M = div.(Ref(N),traj_length)
df = DataFrame(ind=1:100)
ϵ = [0.2]


for m in M
    for eps in ϵ
        addprocs(nprocs)
        @everywhere include("sonar.jl")
        @everywhere include("WR.jl")
        @everywhere using Distributed, DistributedArrays
        @everywhere using Statistics, StatsBase
        println("Running experiments for M = $(m), ϵ = $(eps)...")
        NC,MM,Time = run_exp(N,m,eps,alpha,sonar,mass_mat="identity",silence=false)
        df[!,"eps"*string(Int(eps*100))*"M"*string(m)*"_NC"] = NC
        df[!,"eps"*string(Int(eps*100))*"M"*string(m)*"_MM"] = MM
        df[!,"eps"*string(Int(eps*100))*"M"*string(m)*"_Time"] = Time
        rmprocs(procs()[2:end])
    end
end

CSV.write(file_addr*filename,df)