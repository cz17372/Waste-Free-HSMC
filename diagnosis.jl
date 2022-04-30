using Distributions, DataFrames, CSV, PlotlyJS, StatsPlots

R501 = CSV.read("N10000M50eps10alpha50indentity.csv",DataFrame)
R502 = CSV.read("N10000M50eps20alpha50indentity.csv",DataFrame)
R503 = CSV.read("N10000M50eps30alpha50indentity.csv",DataFrame)
R1001 = CSV.read("N10000M50eps10alpha50indentity.csv",DataFrame)
R1002 = CSV.read("N10000M100eps20alpha50indentity.csv",DataFrame)
R1003 = CSV.read("N10000M100eps30alpha50indentity.csv",DataFrame)
R2001 = CSV.read("N10000M200eps10alpha50indentity.csv",DataFrame)
R2002 = CSV.read("N10000M200eps20alpha50indentity.csv",DataFrame)
R2003 = CSV.read("N10000M200eps30alpha50indentity.csv",DataFrame)


data = R2003
bp1 = box(y=data.MM_sequential,maker_color="grey",name="Sequential")
bp2 = box(y=data.MM_independent,maker_color="indianred",name="Independent")
bp3 = box(y=data.MM_full,maker_color="indiangreen",name="Full")
bp4 = box(y=data.MM_chopin,maker_color="indianyellow",name="Nikola")

PlotlyJS.plot([bp1,bp2,bp3,bp4],Layout(yaxis_range=[0.36,0.48],title="N=10000,M=200,stepsize=0.3",yaxis_title="Mean of Marginals"))

R501 = CSV.read("N50000M50eps10alpha50indentity.csv",DataFrame)
R502 = CSV.read("N50000M50eps20alpha50indentity.csv",DataFrame)
R503 = CSV.read("N50000M50eps10alpha50indentity.csv",DataFrame)
R2001 = CSV.read("N50000M200eps10alpha50indentity.csv",DataFrame)
data = R2001
bp1 = box(y=data.MM_sequential,maker_color="grey",name="Sequential")
bp2 = box(y=data.MM_independent,maker_color="indianred",name="Independent")
bp3 = box(y=data.MM_full,maker_color="indiangreen",name="Full")
bp4 = box(y=data.MM_chopin,maker_color="indianyellow",name="Nikola")
PlotlyJS.plot([bp1,bp2,bp3,bp4],Layout(yaxis_range=[0.36,0.48],title="N=50000,M=200,stepsize=0.1",yaxis_title="Mean of Marginals",boxmode="group"))

data = CSV.read("N10000M500eps30alpha50indentity.csv",DataFrame)
bp1 = box(y=data.NC_sequential,maker_color="grey",name="Sequential")
bp2 = box(y=data.NC_independent,maker_color="indianred",name="Independent")
bp3 = box(y=data.NC_full,maker_color="indiangreen",name="Full")
bp4 = box(y=data.NC_chopin,maker_color="indianyellow",name="Nikola")
PlotlyJS.plot([bp1,bp2,bp3,bp4],Layout(yaxis_range=[-140,-110],title="N=50000,M=500,stepsize=0.3",yaxis_title="Mean of Marginals"))


data = CSV.read("N10000M100eps20alpha50indentity.csv",DataFrame)
bp1 = box(y=data.NC_sequential,maker_color="grey",name="Sequential")
bp2 = box(y=data.NC_independent,maker_color="indianred",name="Independent")
bp3 = box(y=data.NC_full,maker_color="indiangreen",name="Full")
bp4 = box(y=data.NC_chopin,maker_color="indianyellow",name="Nikola")
PlotlyJS.plot([bp1,bp2,bp3,bp4],Layout(yaxis_range=[-140,-110],title="N=10000,M=4,stepsize=0.15",yaxis_title="Log NC"))
