using Plots, StatsPlots, CSV, DataFrames,JLD2
theme(:ggplot2)

dat1 = CSV.read("/Users/changzhang/Desktop/data/orthant/Orthant50000MaxB5.csv",DataFrame)

boxplot(["1000"],dat1[!,2],yticks=(-825:-815),ylims=(-825,-815),label="",xlabel="N",ylabel="LogNC",color=:grey);
boxplot!(["500"],dat1[!,4],label="",color=:grey);
boxplot!(["250"],dat1[!,6],label="",color=:grey);
boxplot!(["100"],dat1[!,8],label="",color=:grey);
boxplot!(["50"],dat1[!,10],label="",color=:grey)
savefig("orthant_lognc.pdf")


boxplot(["1000"],dat1[!,3],yticks=(2.4575:0.0025:2.480),ylims=(2.4575,2.480),label="",xlabel="N",ylabel="Mean of Marginals",color=:grey);
boxplot!(["500"],dat1[!,5],label="",color=:grey);
boxplot!(["250"],dat1[!,7],label="",color=:grey);
boxplot!(["100"],dat1[!,9],label="",color=:grey);
boxplot!(["50"],dat1[!,11],label="",color=:grey)
savefig("orthant_mm.pdf")

dat2 = CSV.read("data/Orthant.csv",DataFrame)
boxplot(NC); @load "data.jld2"
boxplot(dat1[!,4],ylims=(-824,-815),size=(400,800))