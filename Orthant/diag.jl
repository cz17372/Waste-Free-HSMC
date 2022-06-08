using Plots, StatsPlots, CSV, DataFrames,JLD2
theme(:ggplot2)

dat1 = CSV.read("data/Orthant10000MaxB5.csv",DataFrame)

boxplot(["200"],dat1[!,2],yticks=(-825:-815),ylims=(-825,-815),label="",xlabel="M",ylabel="LogNC");
boxplot!(["100"],dat1[!,4],label="");
boxplot!(["50"],dat1[!,6],label="");
boxplot!(["20"],dat1[!,8],label="");
boxplot!(["10"],dat1[!,10],label="")
boxplot!(["1000"],dat2[:,1])


boxplot(["200"],dat1[!,3],yticks=(2.4575:0.0025:2.480),ylims=(2.4575,2.480),label="",xlabel="M",ylabel="Mean of Marginals");
boxplot!(["100"],dat1[!,5],label="");
boxplot!(["50"],dat1[!,7],label="");
boxplot!(["20"],dat1[!,9],label="");
boxplot!(["10"],dat1[!,11],label="")





dat2 = CSV.read("data/Orthant.csv",DataFrame)

boxplot(NC)

@load "data.jld2"

dat1[!,2:2:10]

boxplot(dat1[!,4],ylims=(-824,-815),size=(400,800))