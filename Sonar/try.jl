"""
function Run_Exp(N,M,ϵ,α)
    NC = zeros(100); MM = zeros(100);
    Threads.@threads for n = 1:100
        _,_,_,_,_,NC[n],MM[n]= wasterecycling.SMC(N,M,model=sonar,ϵ=ϵ,α=α,mass_mat="identity")
        println("Simulation $(n): LogNC= ",NC[n]," MM= ",MM[n])
    end
    return NC,MM
end

println("Enter N"); N = readline();
println("Enter the address to store the results");file_addr = readline()
filename = "N"*N*"identityalpha50.csv"
N = parse(Int64,N);
traj_length = [50,100,200,500,1000]
M = div.(Ref(N),traj_length)
df = DataFrame(ind=1:100)
eps = [0.05,0.1,0.2,0.3]
for i = 1:length(M)
    for j = 1:length(eps)
        println("Doing experiment for M = $(M[i]),eps=$(eps[j])")
        NC,MM = Run_Exp(N,M[i],eps[j],0.5)
        df[!,"eps"*string(Int(eps[j]*100))*"M"*string(M[i])*"_NC"] = NC
        df[!,"eps"*string(Int(eps[j]*100))*"M"*string(M[i])*"_MM"] = MM
    end
end

CSV.write(file_addr*filename,df)


@time R = wasterecycling.SMC(10000,100,model=sonar,ϵ=0.2,α=0.2,mass_mat="identity",printl=true);
"""

include("chopin.jl");include("sonar.jl")
R = chopin.SMC(100000,50,model=sonar,α=0.5,printl=true)