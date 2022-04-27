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
N = parse(Int64,N)

R1 = overallexp(N,50,0.1,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M50eps01alpha05indentity.csv"
CSV.write(filename,R1)
R2 = overallexp(N,50,0.2,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M50eps02alpha05indentity.csv"
CSV.write(filename,R2)
R3 = overallexp(N,50,0.3,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M50eps03alpha05indentity.csv"
CSV.write(filename,R3)


R4 = overallexp(N,100,0.1,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M100eps01alpha05indentity.csv"
CSV.write(filename,R4)
R5 = overallexp(N,100,0.2,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M100eps02alpha05indentity.csv"
CSV.write(filename,R5)
R6 = overallexp(N,100,0.3,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M100eps03alpha05indentity.csv"
CSV.write(filename,R6)

R7 = overallexp(N,200,0.1,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M200eps01alpha05indentity.csv"
CSV.write(filename,R7)
R8 = overallexp(N,200,0.2,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M200eps02alpha05indentity.csv"
CSV.write(filename,R8)
R9 = overallexp(N,200,0.3,0.5,sonar,mass_mat="identity",silence=false)
filename = "N"*string(N)*"M200eps03alpha05indentity.csv"
CSV.write(filename,R9)
