from gamspy import Model, Sense, Container, Equation, Variable, Parameter, Set, Sum
import matplotlib.pyplot as mpl
import pandas as pd
import numpy as np

#import data from input.xlsx
input = pd.read_excel("input.xlsx", sheet_name=["Product", "Subassembly", "Scenario"])

#check for the data errors
for _ in input["Product"].to_numpy():
    if (_ < 0).any():
        exit("Invalid data in Product")
for _ in input["Subassembly"].to_numpy():
    if (_ < 0).any():
        exit("Invalid data in Subassembly")
for _ in input["Scenario"].to_numpy():
    if (_ < 0).any():
        exit("Invalid data in Scenario")
if np.sum(input["Scenario"]["P"].to_numpy()) != 1:
    exit("Invalid data in Scenario")

#Build model from data input
m = Container()

i = Set(container=m, name="i", description="list of products", records=input["Product"]["Name"].to_numpy())
j = Set(container=m, name="j", description="list of subassemblies", records=input["Subassembly"]["Name"].to_numpy())
k = Set(container=m, name="k", description="list of scenarios", records=input["Scenario"]["S"].to_numpy())

b = Parameter(container=m, name="b", domain=j, description="cost of subassembly", records=input["Subassembly"]["b"].to_numpy())
s = Parameter(container=m, name="s", domain=j, description="salvage assess", records=input["Subassembly"]["s"].to_numpy())
l = Parameter(container=m, name="l", domain=i, description="cost of product", records=input["Product"]["l"].to_numpy())
q = Parameter(container=m, name="q", domain=i, description="price of product", records=input["Product"]["q"].to_numpy())
A = Parameter(container=m, name="A", domain=[i, j], description="product i requires a_(ij) unit of part j",records=input["Product"].iloc[:,3:].to_numpy())
D = Parameter(container=m, name="D", domain=[k, i], description="demand of product i in scenario s", records=input["Scenario"].iloc[:, 2:].to_numpy())
P = Parameter(container=m, name="P", domain=k, description="probability of scenario s", records=input["Scenario"]["P"].to_numpy())

x = Variable(container=m, name="x", domain=j, description="number of ordered parts", type="positive")
y = Variable(container=m, name="y", domain=[k, j], description="number of left parts", type="positive")
z = Variable(container=m, name="z", domain=[k, i], description="number of products", type="positive")

eq1 = Equation(container=m, name="eq1", domain=[k, j])
eq1[k, j] = y[k, j] == x[j] - Sum(i, A[i, j] * z[k, i])

eq2  = Equation(container=m, name="eq2", domain=[k, i])
eq2[k, i] = z[k, i] <= D[k, i]

ans = Model(container=m, name="ans", equations=[eq1, eq2], problem="LP" , sense=Sense.MIN, objective=Sum(j, b[j] * x[j]) + Sum(k, P[k] * (Sum(i, (l[i] - q[i]) * z[k, i]) - Sum(j, s[j] * y[k, j]))))
#Solve problem
ans.solve()

#plot graphs from the output
fig, ax = mpl.subplots()
k_list = k.records["uni"].to_list()
i_list = i.records["uni"].to_list()
j_list = j.records["uni"].to_list()
size = len(k_list)
ax.bar(j_list, x.records["level"].to_list(), width=0.6)
mpl.title("Number of preordered subassemblies")
mpl.xlabel("Subassembly")
mpl.ylabel("Number")
mpl.savefig("assembly.png")
arr_xy = (1 - y.records["level"].to_numpy().reshape(size, len(j)) / x.records["level"].to_numpy()) * 100
arr_z = z.records["level"].to_numpy() / D.records["value"].to_numpy() * 100
arr_z = arr_z.reshape(size, len(i_list))
for n in range(size):
    fig, ax1 = mpl.subplots()
    ax1.bar(j_list, arr_xy[n], width=0.6)
    mpl.title("Percentage of used subassembly in scenario " + k_list[n])
    mpl.xlabel("Subassembly")
    mpl.ylabel("Percentage")
    mpl.savefig("xy_in_scenario" + k_list[n] + ".png")
    fig, ax2 = mpl.subplots()
    colors = np.where(arr_z[n] > 80, 'green', 'red')
    ax2.bar(i_list, arr_z[n], width=0.6, color=colors)
    mpl.title("Percentage of demand satisfied in scenario " + k_list[n])
    mpl.xlabel("Product")
    mpl.ylabel("Percentage")
    mpl.savefig("z_in_scenario" + k_list[n] + ".png")

#write output data to output.xlsx
df1 = pd.DataFrame(x.records)
df2 = pd.DataFrame(y.records)
df3 = pd.DataFrame(z.records)
df4 = pd.DataFrame([ans.objective_value], index=["Objective Value"])
with pd.ExcelWriter("output.xlsx") as writer:
    df1.to_excel(writer, sheet_name="order")
    df2.to_excel(writer, sheet_name="inventory")
    df3.to_excel(writer, sheet_name="produce")
    df4.to_excel(writer, sheet_name="objective", header=False)