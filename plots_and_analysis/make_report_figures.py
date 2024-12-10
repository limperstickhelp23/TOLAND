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

def style(alpha=.05,rows=1,cols=1,num_colors=5): 
    fig,ax=plt.subplots(rows,cols,dpi=150)
    ax.patch.set_facecolor("lightgrey")
    ax.patch.set_alpha(alpha)
    cmap = plt.cm.viridis  # Choose a colormap (e.g., 'viridis', 'plasma', etc.)
    colors = cmap(np.linspace(0, 1, len(data)))  # Create gradient colors

    ax.set_ylabel("Global Model Accuracy",fontweight="bold",fontsize=12)
    ax.set_xlabel("Server Rounds",fontweight="bold",fontsize=12)
    
    return fig,ax,colors

def training_curves(data):
    """  NOTE: plots the core algorithm ideas:  cosine, ppa, dpp, and star baseline
    """
    # Plot each dataset with a different color from the gradient
    fig,ax,colors=style(alpha=.15)

    for (n,(name,series)) in enumerate(data.items()):      

        if name == "maxdpp":
            width=2
        elif name == "star":
            width=3
        else:
            width=1.5

        ax.plot(series, label=name.upper(), marker= "x", markersize=3, color=colors[n], linewidth=width)

    fig.suptitle("Classifier Performance (CIFAR-10)", fontsize=16)
    plt.legend(),plt.grid()
    # plt.show()
    plt.savefig("report/milestone_3_training_curves.png")




def mnist_round_over_round_cost_comparison():
    data={"accuracies":{}, "losses":{},"costs":{}, "clocks":{},"norm_costs":{}, "norm_clocks":{}}
    for name in os.listdir("mnist_cost_analysis/"):
        with open(os.path.join("mnist_cost_analysis/",name), "r") as file:
            d = json.load(file)
            data["accuracies"][name]=d['cloud']['accuracies']
            b=np.array( [0] + data["accuracies"][name])
            a=np.diff(b)
            print(a,b)
            denom=a
            data["losses"][name]=d['cloud']['losses']
            data["costs"][name]=d["server_round_costs"]
            if name != "star.json":
                data["costs"][name]=data["costs"][name][::2] #NOTE: hack to fix a stroing bug I just noticed with duplicates
            print(name)
            print(data["costs"][name])
            print(denom)
            data["clocks"][name]=d["cloud"]["Wall_Clock"]
            data["norm_clocks"][name]=list(np.array(data["clocks"][name])/denom)
            data["norm_costs"][name]=list(np.array(data["costs"][name])/denom)
    
    def pad_mnist_metric(series):
        max_length=8
        last_value = series[-1]  # Get the last value of the series
        return series + [last_value] * (max_length - len(series))

    
    fig,ax=plt.subplots(2,2,dpi=250,figsize=(10,7))
    cmap = plt.cm.tab10 # Choose a colormap (e.g., 'viridis', 'plasma', etc.) or tab10
    colors = cmap(np.linspace(0, .5, len(data['accuracies'])))  # Create gradient colors

    for (i,j) in ([0,0],[1,1],[1,0],[0,1]):
        # ax[i,j].set_xlabel("Server Rounds",fontweight="bold",fontsize=10)
        ax[i,j].patch.set_facecolor("lightgrey")
        ax[i,j].patch.set_alpha(.15)
        ax[i,j].grid()
    
    print(data['accuracies'].keys())
    
    labels=[ (name[:-5]).upper() for name in data["accuracies"].keys()]
    titles=["Network Communication Cost", "Wall Clock Runtime (s)", "Normalized", "Normalized"]


    ## NOTE: raw cost metrics
    for (m,metric) in enumerate( (data['costs'], data['clocks'], data["norm_costs"], data["norm_clocks"])):
        i,j=m//2,m%2
        ax[i,j].set_title(titles[m],fontweight='bold',fontsize=12,style='italic')
        for (n,(name,series)) in enumerate(metric.items()):
            x=list(np.cumsum(series))       
            ax[i,j].plot(pad_mnist_metric(x), color=colors[n], label=labels[n], linewidth=2)


    ax[0,0].legend(loc='upper left')
    fig.suptitle("Cost Metric Comparisons:\n 8 Server Rounds", fontsize=16)
    fig.tight_layout()
    # plt.show()
    plt.savefig("report/cost_metric_comparisons.png")

    return


def fedmax_fedprox_comparison():
    
    # results=namedtuple('results',["accuracies","losses", "cost"])
    data={"accuracies":{}, "losses":{},"costs":{}}

    for name in ["maxdpp2.json", "proxdpp.json"]:
        with open(os.path.join(FOLDER,name), "r") as file:
            d = json.load(file)
            data["accuracies"][name]=d['cloud']['accuracies']
            data["losses"][name]=d['cloud']['losses']
            data["costs"]=d["total_run_cost"]
        

    fig,ax=plt.subplots(1,2,dpi=200,figsize=(11,6))
    cmap = plt.cm.viridis  # Choose a colormap (e.g., 'viridis', 'plasma', etc.)
    colors = cmap(np.linspace(0, 1, len(data)))  # Create gradient colors

    for a in ax:
        a.set_xlabel("Server Rounds",fontweight="bold",fontsize=12)
        a.patch.set_facecolor("lightgrey")
        a.patch.set_alpha(.15)
        a.grid()

    labels=["FedMax DPP", "FedProx DPP"]
    titles=["Global Accuracy", "Global Loss"]

    for (ii,metric) in enumerate( (data['accuracies'], data['losses'])):
        ax[ii].set_title(titles[ii],fontweight="bold",fontsize=10)
        for (n,(name,series)) in enumerate(metric.items()):
            ax[ii].plot(series, color=colors[n], label=labels[n])

    plt.legend(), 
    fig.suptitle("FedMax DPP v. Fed Prox DPP \n 60 Server Rounds", fontsize=16)
    fig.tight_layout()
    plt.savefig("report/maxprox_comparison.png")
    return


def wall_clock():
    clocks={}
    for name in NAMES:
        with open(os.path.join(FOLDER,name+".json"), "r") as file:
            d = json.load(file)
            clocks[name]=np.sum(d['cloud']['Wall_Clock'])
    return clocks


def report_total_costs():
    """NOTE: normalized cost metrics for Table 2 in final report
        "Normalized" means -> avg cost required to reach 46% which was the best star could do in 35 rounds (others reached this faster)
    """
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


if __name__ == "__main__":

    fedmax_fedprox_comparison()
    training_curves(data)
    report_total_costs()
    mnist_round_over_round_cost_comparison()

    