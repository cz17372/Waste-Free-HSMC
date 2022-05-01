using CSV, DataFrames, StatsPlots, PlotlyJS
data_M4  = CSV.read("data/LR/Chopin/N10000M4eps1080identitychopin.csv",DataFrame)
data_M10 = CSV.read("data/LR/Chopin/N10000M10eps10alpha80identitychopin.csv",DataFrame)
data_M50 = CSV.read("data/LR/Chopin/N10000M50eps20alpha80identitychopin.csv",DataFrame)

f = Figure()
ax = Axis(f[1,1],ylabel=L"Log L_T",xticks=([1,2,3],["M=4","M=50","M=100"]))
theme(:ggplot2)
boxplot(repeat(["4","10","50"],inner=100),vcat(data_M4.NC_chopin,data_M10.NC_chopin,data_M50.NC_chopin),label="",ylim=(-140,-110),)

data_M10[!,"newcol"] = rand(100)

data_M10


using CSV, DataFrames, StatsPlots, PlotlyJS

data = CSV.read("data/sonar/full.csv",DataFrame)
bp1 = box(y=data.N10000M200eps20alpha50_NC)
PlotlyJS.plot(bp1,Layout(yaxis_range=(-140,-110)))