import os
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
NAMES=["maxdpp","cosine","ppa","dpp", "star"]
FOLDER="cifar_m3_final"

data={}
nets={}

for name in NAMES:
    with open(os.path.join(FOLDER,name+".json"), "r") as file:
        d = json.load(file)
        data[name]=d['cloud']['accuracies']
        nets[name]=d["total_run_cost"]

def training_curves(data):

    cmap = plt.cm.viridis  # Choose a colormap (e.g., 'viridis', 'plasma', etc.)
    colors = cmap(np.linspace(0, 1, len(data)))  # Create gradient colors

    # Plot each dataset with a different color from the gradient
    fig,ax=plt.subplots(dpi=150)
    ax.patch.set_facecolor("lightgrey")
    ax.patch.set_alpha(.05)

    for (n,(name,series)) in enumerate(data.items()):      

        if name == "maxdpp":
            width=2
        else:
            width=1.5
    
        plt.plot(series, label=name.upper(), marker= "x", markersize=3, color=colors[n], linewidth=width)

    plt.grid()
    plt.ylabel("Global Model Accuracy",fontweight="bold",fontsize=12)
    plt.xlabel("Server Rounds",fontweight="bold",fontsize=12)
    plt.title("Classifier Performance (CIFAR-10)", fontsize=16)
    plt.legend()
    plt.savefig("milestone_3.png")

def wall_clock():
    clocks={}
    for name in NAMES:
        with open(os.path.join(FOLDER,name+".json"), "r") as file:
            d = json.load(file)
            clocks[name]=np.sum(d['cloud']['Wall_Clock'])

    # print(clocks)
    return clocks

# training_curves(data)

totals=wall_clock()

norm_clocks={}
norm_nets={}
for (k,v) in totals.items():
    avg=v/35
    rounds=np.argmax( np.array(data[k]) > 0.4556) 
    # netcost=d[k]["total_run_cost"]
    if k == "star":
        norm_clocks[k] = totals[k]
        norm_nets[k] = nets[k]
    else:
        norm_clocks[k] = avg*rounds
        norm_nets[k] = (nets[k]/35)*rounds
        print(nets[k])
print(norm_clocks)
print(norm_nets)