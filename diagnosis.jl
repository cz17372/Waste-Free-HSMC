using CSV, DataFrames, StatsPlots,LaTeXStrings
theme(:ggplot2)
data_adaptive = CSV.read("data/sonar/full4.csv",DataFrame)
data_fix      = CSV.read("data/sonar/full_final.csv",DataFrame)

M = [50,150,200,250,300,400,500,600]
name = Vector{String}(undef,length(M))
for n = 1:length(M)
    name[n] = string(div(30000,M[n])*0.05)
end
function df_to_mat(df)
    N,D = size(df)
    mat = zeros(N,D)
    for n = 1:D
        mat[:,n] = df[:,n]
    end
    return mat
end
function plotbp(name,df;args...)
    _,N = size(df)
    boxplot([name[1]],df[:,1],color=:gray,leg=false;args...)
    for n = 2:N
        boxplot!([name[n]],df[:,n],leg=false,color=:gray)
    end
    current()
end
eps = 0.05
name = Vector{String}(undef,length(M))
for n = 1:length(M)
    name[n] = string(div(30000,M[n])*eps)
end
plotbp(name,df_to_mat(data_fix[!,r"eps"*string(Int(eps*100))][!,r"NC"]),ylim=(-140,-110),title=L"\epsilon ="* "$(eps)",xlabel=L"\tau")

boxplot(df_to_mat(data_adaptive[:,3:3:end]),leg=false,color=:gray,ylim=(-140,-110))


boxplot(df_to_mat(data_fix[!,r"eps25"][!,r"NC"]),leg=false,color=:gray,ylim=(-140,-110))

[name[1]]
name = ["0.10","0.15","0.20","0.25","0.30"]
fixtau_15 = df_to_mat(data_fix[!,["N30000M200eps10alpha50_NC","N30000M300eps15alpha50_NC","N30000M400eps20alpha50_NC","N30000M500eps25alpha50_NC","N30000M600eps30alpha50_NC"]])
plotbp(name,fixtau_15,ylim=(-140,-110),title=L"\tau ="* "15",xlabel=L"\epsilon")

name = ["0.05","0.15","0.20","0.25","0.30"]
fixtau_30 = df_to_mat(data_fix[!,["N30000M50eps5alpha50_NC","N30000M150eps15alpha50_NC","N30000M200eps20alpha50_NC","N30000M250eps25alpha50_NC","N30000M300eps30alpha50_NC"]])
plotbp(name,fixtau_30,ylim=(-140,-110),title=L"\tau ="* "30",xlabel=L"\epsilon")

name = ["0.10","0.30"]
fixtau_60 = df_to_mat(data_fix[!,["N30000M50eps10alpha50_NC","N30000M150eps30alpha50_NC"]])
plotbp(name,fixtau_60,ylim=(-140,-110),title=L"\tau ="* "60",xlabel=L"\epsilon")