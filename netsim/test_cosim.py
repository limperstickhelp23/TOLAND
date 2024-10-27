# from netsim.network_model import MobileNet
from netsim.network import MobileNet
# from data_prepare import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from algorithms import *
import matplotlib.pyplot as plt
import torch
from random import sample

GREEN = "\033[92m"   # Bright Green
YELLOW = "\033[93m"  # Bright Yellow
RED = "\033[91m"     # Bright Red
RESET = "\033[0m"    # Reset to default color

##NOTE Plotting Colors
COLORS=[
    "red","blue","gold","green","lavender","magenta","orange","grey","firebrick","brown","tab:blue","darkgreen","indigo"
    "black", "teal", "bisque", "mediumturquoise", "darkviolet"
]

#UTIL
def check_similarity(algorithm,gt):
        for d in algorithm.device_list:
            if d.parent_point != gt[d.id]:
                print(f"Incorrect Asisgnment {d.id}, {d.parent_point}\nCorrect Assignment: {gt[d.id]}\nPossible Assignments:")
                for ap in algorithm.curr_apoints:
                    print(f"AP: {ap}, Cosine Sim: {compute_device_to_community_cosim(d, algorithm.device_list[ap])}")
                print()

# print(d.id, " " , d.parent_point, ":\n ", d.community_model.state_dict()['conv1.bias'], "\n", d.model.state_dict()['conv1.bias'])
class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compute_cosim_metric(m1: nn.Parameter, m2: nn.Parameter):
    sim,cos=0, nn.CosineSimilarity(dim=0, eps=1e-6)
    for (p1,p2) in zip(m1,m2):
        layer_sim=abs(cos(p1.data.flatten(),p2.data.flatten()).item())
        sim += layer_sim
    return sim

def compute_device_model_cosim_metric(dvc1, dvc2):
    m1=dvc1.model.parameters()
    m2=dvc2.model.parameters()
    return compute_cosim_metric(m1,m2)

def compute_device_to_community_cosim(dvc1, dvc2):
    m1=dvc1.model.parameters()
    m2=dvc2.community_model.parameters()
    return compute_cosim_metric(m1,m2)

def plot_colored_graph(G,colors,pos,ax=None,access=[], spring=False):

    if (spring==True):
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    if ax==None:
        fig,ax=plt.subplots()

    for node_color, nodelist in colors.items():
        for node in nodelist:
            node_size,node_shape=230,'o'
            if node in access:
                node_size,node_shape=1000,'*'
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color,node_size=node_size,node_shape=node_shape,ax=ax)
        labels = {x: x for x in G.nodes}

    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='w',ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(),width=.5, alpha=0.75,ax=ax)


def three_comm_test_with_aggregation(
        size=10,threshold=25,movement=False,plot_result=False, spring_plot=False,
        print_final=True, print_sims = False
    ):

    cosine=CosineReassignment(num_devices=size)
    cosine.threshold=threshold
    cosine.num_apoints=3
    dummies = [
        ("red",0),
        ("blue",1),
        ("gold",2)
    ]
    gt_nodes={
        "red":[],
        "blue":[],
        "gold":[]
    }
    invert_gt_nodes={}

    cosine.build_proximity_graph()
    cosine.select_access_points_on_betweenness()
    gt = {ap:[] for ap in cosine.curr_apoints}

    # NOTE: Important INIT STEP
    """Go through all, and then re-assign access points"""
    ap_color_index=0
    for (n,d) in enumerate(cosine.device_list):
        color,index = dummies[n%3]
        
        if d.id in cosine.curr_apoints:
            index=ap_color_index%3
            color=COLORS[ap_color_index]
            gt[d.id]=d.id
            ap_color_index+=1

        d.color=color
        gt_nodes[color].append(n)
        invert_gt_nodes[n]=color
        
        for key in d.model.state_dict().keys():
            d.model.state_dict()[key][index]=100

    for (n,d) in enumerate(cosine.device_list):
        gt[n]=d.id
        for ap in cosine.curr_apoints:
            if d.color == cosine.device_list[ap].color:
                gt[n]=ap
    cosine.assign_communities()
    cosine.init_community_models()
    initial={COLORS[i%len(COLORS)]: vals for (i,vals) in enumerate(cosine.ap_member_map.values())}
    after=initial
    #NOTE: ensure each access point knows it's GT color
    for dvc in cosine.device_list:
        dvc.color=invert_gt_nodes[dvc.id]

    ## NOTE: Start Simulation
    for rnd in range(5):
        check=np.sum([a == b for a,b in zip(list(after.values()),list(gt_nodes.values()))])
        if (check == 3):
            break
        ## "Retrain" the signal
        for (n,d) in enumerate(cosine.device_list):
            index=n%3
            if n in cosine.curr_apoints:
                continue

            d.model.state_dict()['conv1.bias']=torch.zeros_like(d.model.state_dict()['conv1.bias'])
            for key in d.model.state_dict().keys():
                d.model.state_dict()[key]=torch.zeros_like(d.model.state_dict()[key])
                d.model.state_dict()[key][index]=100

        cosine.compare_communities(max_iters=3)
        after={cosine.device_list[k].color:vals for (k,vals) in cosine.ap_member_map.items()}
        cosine.local_aggregation_round()

        if (check < 3) and (print_sims):
            print(f"\nROUND {rnd}\n")
            check_similarity(cosine, gt=gt)

    if plot_result:
        fig,ax=plt.subplots(1,3,figsize=(15,5))
        plot_colored_graph(cosine.NXG1, initial, pos=cosine.get_positions(),ax=ax[0], access=cosine.curr_apoints, spring=spring_plot)
        ax[0].set_title("Before Self-Assignment")
        plot_colored_graph(cosine.NXG1, after, pos=cosine.get_positions(),ax=ax[1],access=cosine.curr_apoints,spring=spring_plot)
        ax[1].set_title("After Self-Assignment")
        plot_colored_graph(cosine.NXG1, gt_nodes, pos=cosine.get_positions(),ax=ax[2],access=cosine.curr_apoints,spring=spring_plot)
        ax[2].set_title("Ground Truth Solution")
        fig.tight_layout()
    
    if (print_final):
        print(f"Check Matches: \n {list(gt_nodes.values())} \n {list(after.values())}")
    
    return (check==3)

def output(count,num_trials):
    if count == num_trials:
        print(f"{GREEN}All tests passed! {RESET}")
    elif count > 0 and count < num_trials:
        print(f"{YELLOW}Some tests failed. {(count)*100/num_trials} % pass rate {RESET}")
    else:
        print(f"{RED}No tests passed. ❌ {RESET}")

def community_cosine_fully_connected_test(num_trials=100,num_devices=9):
    count=0
    for _ in range(num_trials):
        count+=three_comm_test_with_aggregation(size=num_devices,plot_result=False,print_final=False)
    output(count,num_trials)

def community_cosine_disconnected_test(num_trials=100,threshold=7,num_devices=27):
    count=0
    for _ in range(num_trials):
        count+=three_comm_test_with_aggregation(size=num_devices,threshold=threshold,plot_result=False, print_final=False)
    output(count,num_trials)


if __name__ == "__main__":
    #NOTE: Fully Connected Case -- Should Have 100% pass rate
    # In first test rare instances they were getting pulled into wrong community but that was just property of noise in cosine metric
    community_cosine_fully_connected_test(num_trials=0)  #NOTE: INCREASE to 100

    #NOTE: Proximity Case -- Not sure what will happen yet
    community_cosine_disconnected_test(num_trials=0)  #NOTE: INCREASE to 100

    #NOTE: Visualize Fully Connected
    three_comm_test_with_aggregation(size=20,threshold=100,plot_result=True,spring_plot=True)
    plt.suptitle("Fully Connected Finds All the Neighbors")
    plt.tight_layout()

    ## NOTE: Visualize Partial Connections
    three_comm_test_with_aggregation(size=20,threshold=10,plot_result=True,spring_plot=True, print_final=True)
    plt.suptitle("Partially Connected is Mixed: Far Away Nodes Do Not Receive Signal")
    plt.tight_layout()
    
    
    plt.show()
