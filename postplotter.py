import os
import matplotlib.pyplot as plt
import numpy as np

# RUN="metrics/run_5"
## NOTE: this is very manual right now and can be greatly improved.
## TODO: also probably move this to utils file

RUN ="run_14"
TITLE = "Small World"

RUN ="run_11"
TITLE = "Random (ER)"

RUN ="run_10"
TITLE = "Scale Free"

getacc=lambda s: s[s.find("accuracy:")+9:].strip(" ")
getround=lambda s: int(s[len("round"):s.find(":")].strip(""))
getap=lambda s: int(s[:s.find(":")].strip(""))


def sort_dictionary(d,desc=True,key=0):
    return sorted(d.items(), key = lambda item: item[key], reverse=desc)

METS=sorted(os.listdir(RUN))
print(METS)
GLOB=os.path.join(RUN,METS.pop())
METS=sorted(METS, key=lambda s: int(s.split('_')[-1]))
print(METS)


with open(GLOB, "r") as f:
    x=[line.strip("\n") for line in f.readlines()]


def parse_ap_data():
    apoints, key = {}, 0

    for eval in METS:
        local=sorted(os.listdir(os.path.join(RUN,eval))).pop()
        with open(os.path.join(RUN,eval,local), "r") as f:
            y=f.readlines()

        for (n,line) in enumerate(y):
            if n%3 == 0:
                key=getap(line)
                if key not in apoints.keys():
                    apoints[key]=[]
            if n%3 == 2:
                a=float(getacc(line))
                apoints[key].append(a)
            
    return apoints

def star():
    run ="run_13/global_model_eval.txt"
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
    
apoints=parse_ap_data()
star_baseline = star()

accuracies = {}
round = 0
for line in x:
    line.strip()
    
    if "round" in line:
        round=getround(line)
        accuracies[round] = 0.0
    if "accuracy" in line:
        accuracies[round]=(float(getacc(line)))
    
rounds,globs=zip(*sort_dictionary(accuracies, desc=False, key=0))
srounds,stars=zip(*sort_dictionary(star_baseline, desc=False, key=0))
stars=list(stars)
globs=list(globs)

stars.insert(0,0)
if RUN == "run_10":
    print(len(stars))
    diff = len(stars)-len(globs)
    stars = stars[:len(globs)]

else:
    diff = len(globs)-len(stars)
    stars.extend([stars[-1]]*(diff))


# Set up a colormap
cmap = plt.cm.viridis  # Choose a colormap (e.g., 'viridis', 'plasma', etc.)
colors = cmap(np.linspace(0, 1, len(apoints)))  # Create gradient colors

# Plot each dataset with a different color from the gradient
fig,ax=plt.subplots(dpi=200)
ax.patch.set_facecolor("lightgrey")
ax.patch.set_alpha(.05)
for ((ap, values), color) in zip(apoints.items(), colors):
    plt.plot(rounds, values, label="AP Device: "+str(ap), marker= "x", markersize=5, color=color, linewidth=1.5,)
plt.plot(rounds,globs, color="red", linewidth = 2.5, marker = "x", label="Global")
plt.plot(rounds,stars, color="magenta",label="Central Baseline",linewidth = 2.0, marker = "x")
plt.grid()

plt.ylabel("Test Accuracy",fontweight="bold",fontsize=12)
plt.xlabel("Global Aggregation Rounds",fontweight="bold",fontsize=12)
plt.title(TITLE, fontsize=20)
plt.legend()
plt.show()

