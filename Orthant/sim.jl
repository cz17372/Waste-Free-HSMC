include("SMC.jl")
using JLD2; @load "Orthant/Sigma.jld2"
using DataFrames,CSV
function Run_Exp(N,M,MaxB,Σ,a,b)
    NC = zeros(100); MM = zeros(100);
    Threads.@threads for n = 1:100
        NC[n],MM[n]= SMC(N,M,0.2,Σ,a,b,150,MaxBounces=MaxB)
        println("Simulation $(n): LogNC= ",NC[n]," MM= ",MM[n])
    end
    return NC,MM
end
println("Enter N"); N = readline();
println("Enter the maximum number of bounces"); MaxB = readline()
println("Enter the address to store the results");file_addr = readline()
filename = "N"*N*"MaxB"*MaxB*".csv"
N = parse(Int64,N); MaxB = parse(Float64,MaxB)
traj_length = [50,100,200,500,1000]
M = div.(Ref(N),traj_length)
a = 1.5*ones(150); b = Inf*ones(150)
df = DataFrame(ind=1:100)
for i = 1:length(M)
    println("Doing experiment for M = $(M[i])")
    NC,MM = Run_Exp(N,M[i],MaxB,Σ,a,b)
    df[!,"NC_"*string(M[i])] = NC
    df[!,"MM_"*string(M[i])] = MM
end
CSV.write(file_addr*filename,df)