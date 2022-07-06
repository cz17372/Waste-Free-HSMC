using Plots, StatsPlots, CSV, DataFrames,JLD2,Measures,LaTeXStrings
theme(:ggplot2)

dat1 = CSV.read("/Users/changzhang/Desktop/data/Sonar/N10000alpha50withtime.csv",DataFrame)
dat2 = CSV.read("/Users/changzhang/Desktop/data/Sonar/ChopinN10000alpha50withtime.csv",DataFrame)
boxplot(["Hamiltonian Snippet SMC"],dat1[!,"eps20M100_Time"],label="",color=:grey,ylabel="Simulation Time (Seconds)",ylims=(0,12))
boxplot!(["waste-free SMC"],dat2[!,"M100_Time"],label="",color=:grey)
savefig("Sonar_SimTime.pdf")

ylim = (0.36,0.48); tp = "MM"; ytitle="Mean of Marginals"
p1 = boxplot(["10"],dat1[!,"eps20M10_"*tp],ylims=ylim,label="",color=:grey,xlabel="N",ylabel=ytitle,title=L"\epsilon = 0.2");
boxplot!(["20"],dat1[!,"eps20M20_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["50"],dat1[!,"eps20M50_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["100"],dat1[!,"eps20M100_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["200"],dat1[!,"eps20M200_"*tp],ylims=ylim,label="",color=:grey)
p2 = boxplot(["10"],dat1[!,"eps5M10_"*tp],ylims=ylim,label="",color=:grey,xlabel="N",ylabel=ytitle,title=L"\epsilon = 0.05");
boxplot!(["20"],dat1[!,"eps5M20_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["50"],dat1[!,"eps5M50_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["100"],dat1[!,"eps5M100_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["200"],dat1[!,"eps5M200_"*tp],ylims=ylim,label="",color=:grey)
p3 = boxplot(["10"],dat1[!,"eps10M10_"*tp],ylims=ylim,label="",color=:grey,xlabel="N",title=L"\epsilon = 0.1");
boxplot!(["20"],dat1[!,"eps10M20_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["50"],dat1[!,"eps10M50_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["100"],dat1[!,"eps10M100_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["200"],dat1[!,"eps10M200_"*tp],ylims=ylim,label="",color=:grey)
p4 = boxplot(["10"],dat1[!,"eps30M10_"*tp],ylims=ylim,label="",color=:grey,xlabel="N",title=L"\epsilon = 0.3");
boxplot!(["20"],dat1[!,"eps30M20_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["50"],dat1[!,"eps30M50_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["100"],dat1[!,"eps30M100_"*tp],ylims=ylim,label="",color=:grey);
boxplot!(["200"],dat1[!,"eps30M200_"*tp],ylims=ylim,label="",color=:grey)

p5 = boxplot(["10"],dat2[!,"M10_"*tp],ylims=ylim,label="",color=:grey,xlabel="N");
boxplot!(["20"],dat2[!,"M20_"*tp],ylims=ylim,label="",color=:grey,xlabel="N");
boxplot!(["50"],dat2[!,"M50_"*tp],ylims=ylim,label="",color=:grey,xlabel="N");
boxplot!(["100"],dat2[!,"M100_"*tp],ylims=ylim,label="",color=:grey,xlabel="N");
boxplot!(["200"],dat2[!,"M200_"*tp],ylims=ylim,label="",color=:grey,xlabel="N")



cp1 = plot(p2,p3,p1,p4,layout=(2,2),size=(600,600),margin=1pt)
cp2 = plot(p5,layout=(1,1),size=(600,600),ylabel=ytitle)
plot(cp1,cp2,layout=(1,2),size=(1200,600),margin=15pt)
savefig("Sonar_MM_alpha70.pdf")