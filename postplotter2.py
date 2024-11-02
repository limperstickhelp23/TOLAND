import os
import matplotlib.pyplot as plt
import numpy as np
import json


FILE="results.json"

with open("results.json", "r") as file:
    data = json.load(file)

getacc=lambda s: s[s.find("accuracy:")+9:].strip(" ")
getround=lambda s: int(s[len("round"):s.find(":")].strip(""))
getap=lambda s: int(s[:s.find(":")].strip(""))
def sort_dictionary(d,desc=True,key=0):
    return sorted(d.items(), key = lambda item: item[key], reverse=desc)  

def star():
    run ="metrics/run_13/global_model_eval.txt"
    round=0
    accuracies = {}
    with open(run, "r") as f:
        x=[line.strip("\n") for line in f.readlines()]
    for line in x:
        line.strip()
        if "round" in line:
            round=getround(line)
            accuracies[round] = 0.0
        if "accuracy" in line:
            accuracies[round]=(float(getacc(line)))
    return accuracies

star_baseline = star()
cloudres=data["cloud"]
globs=cloudres["accuracies"]
del(data["cloud"])
# localres={int(key): data[key] for key in data.keys()}
rounds=[int(key) for key in data.keys()]
apoints=data[str(0)][str(0)]["ap_nodes"]
d = [round_data for round_data in data.values()]

# "0th Round" -- Still need to think about how to visualize this -- maybe separate
loss_matrix = [round_data['0']["losses"] for round_data in data.values()]
acc_matrix = [round_data['0']["accuracies"] for round_data in data.values()]
acc_matrix_transposed = list(zip(*acc_matrix))

cmap = plt.cm.viridis  # Choose a colormap (e.g., 'viridis', 'plasma', etc.)
colors = cmap(np.linspace(0, 1, len(apoints)))  # Create gradient colors

# Plot each dataset with a different color from the gradient
fig,ax=plt.subplots(dpi=150)
ax.patch.set_facecolor("lightgrey")
ax.patch.set_alpha(.05)

for (n,series) in enumerate(acc_matrix_transposed):   
   
    plt.plot(rounds,series,
        label="AP Device: "+str(apoints[n]), marker= "x", markersize=5, color=colors[n], linewidth=1.5,
    )

srounds,stars=zip(*sort_dictionary(star_baseline, desc=False, key=0))
stars=list(stars)
# for ((ap, values), color) in zip(apoints.items(), colors):
#     plt.plot(rounds, values, label="AP Device: "+str(ap), marker= "x", markersize=5, color=color, linewidth=1.5,)
#     plt.plot(rounds,)

plt.plot(rounds,globs, color="red", linewidth = 2.5, marker = "x", label="Global")
# plt.plot(rounds,stars, color="magenta",label="Central Baseline",linewidth = 2.0, marker = "x")
plt.grid()

plt.ylabel("Test Accuracy",fontweight="bold",fontsize=12)
plt.xlabel("Global Aggregation Rounds",fontweight="bold",fontsize=12)
plt.title("Summary", fontsize=20)
plt.legend()
plt.show()

