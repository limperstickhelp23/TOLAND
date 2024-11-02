import networkx as nx
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from itertools import combinations
import yaml, random
import matplotlib.cm as cm
import json
from torch import nn
import torch.nn.functional as F
import torch
from typing import List

# NOTE: manual color map | use built-ins from matplotlib
COLORS=[
    "red","blue","gold","green","lavender","magenta","orange","grey","firebrick","brown","tab:blue","darkgreen","indigo"
    "black", "teal", "bisque", "mediumturquoise", "darkviolet"
]

# Utilities
def get_sampled_colors(n, colormap='viridis'):
    cmap = cm.get_cmap(colormap, n)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    random.shuffle(colors)
    return colors

def sort_dictionary(d,desc=True):
    return sorted(d.items(), key = lambda item: item[1], reverse=desc)

def betweeness_rule(G: nx.Graph, top = 10, weight=None):
    return zip(*sort_dictionary(nx.betweenness_centrality(G,weight=weight))[:top])

def aggregate_params(model_params):
    averaged_state_dict = {}

    for key in model_params[0].keys(): #conv1_
        param_stack = torch.stack([state_dict[key] for state_dict in model_params], dim=0)
        avg_params = torch.mean(param_stack, dim=0)
        averaged_state_dict[key] = avg_params

    return averaged_state_dict

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

class Device:
    def __init__(self,id=0, x_max=100,y_max=100, λ=10,type=[4,5], num_classes=10):
        self.x=np.random.randint(-y_max,y_max)
        self.y=np.random.randint(-x_max,x_max)
        self.λ=λ
        self.id=id
        self.model=Net(num_classes)
        self.community_model=Net(num_classes)  #NOTE: really these are community parameters, after a global round they are the global params
        self.color='red'
        self.parent_point=0 #NOTE: not sure what a good deafault is (like -- the server maybe)
        #self.community = (0,"red")
        
        """NOTE
            Device stores a copy of the community it is assigned to in the parent point attribute
        """

    def set_lambda(self,λ):
        self.λ=λ
    
    def select_distance(self):
        return np.random.exponential(scale=5)
    
    def new_coordinate(self):
        r=self.select_distance()
        θ=np.random.uniform(0,2*pi)
        self.x=self.x+r*np.cos(θ)
        self.y=self.y+r*np.sin(θ)
    
    def compute_cosines():
        return

class MobileNet:
    def __init__(self,num_devices=10,num_classes=10,perceptual_map="rainbow",λ=10):
        
        self.N=num_devices # self.devices={}
        self.device_list=[Device(i,λ=λ,num_classes=num_classes) for i in range(self.N)]
        self.D = np.zeros([self.N,self.N])
        self.A = np.zeros([self.N,self.N])
        self.d_max = 25
        self.curr_apoints = []
        self.num_apoints = 5
        self.ap_color="black"
        self.reset_topologies()
        self.topology_cost=0
        self.AGG_COST = 100
        self.assignments = {} #NOTE: this might be legacy -- directly corresponds to the old file structure (AP, route: etc.)
        self.ap_member_map = {} #NOTE: key AP: values list of everyone assigned to it
        self.routes = {}
        self.ap_params = {}  #NOTE/TODO: store parameters in sim object, then devices can look up parameters
        # self.server=Device() #TODO/ Store here ?
        # self.server.x,self.server.y = 98,98
        self.GEN=GraphGenerator(num_devices)
        self.threshold = 10
        self.colormap = perceptual_map
        self.cmap=get_sampled_colors(n=self.N,colormap=perceptual_map) # TODO make even perceptual spacing dynamically at init
        self.reset_colors()
        """
            TODO: Legacy stuff
                Need to clean up some of the data structures. Assignments is left over from making yaml files accoriding to "route"

            NOTE:
            Instead, try separating out:
                - memberships: key= access point , values = everyone currently assigned to that point
                - routes: store the actual path to AP for calculating communication cost
                - assignments: key: device id, value: parent point id ( sometimes is convenient)  TODO: remove and use Device attribute instead
            Piece together the yaml files from these attributes however they should work
        """
    
    def Euclidean(self,d1:Device, d2:Device):
        return np.sqrt(
            np.linalg.norm( np.array([d1.x,d1.y]) - np.array([d2.x,d2.y]))
        )
    
    def reset_topologies(self):
        self.A = np.zeros([self.N,self.N])
        self.NXG1 = nx.from_numpy_array(self.A) # sparse communities
        self.NXG2 = nx.from_numpy_array(self.A)
        self.topology_cost = 0

    def reset_colors(self, ap_color=None):
        # Assign/ populate colors from the ap_member_map and matpltlib color_map
        self.colors={self.cmap[n]: v for (n,v) in enumerate(self.ap_member_map.values())}
        if ap_color is not None:
            self.colors[ap_color]=self.curr_apoints

    def init_server(self,pos=[0,0]):
        self.server = Device()
        self.server.x = pos[0]
        self.server.y = pos[1]
        return
    
    def set_er_topology(self, p=.05,num_edges=100,allow_isolates=False):
        self.reset_topologies()
        self.GEN.erdos_renyi(p,num_edges,allow_isolates)
        self.NXG1=nx.from_numpy_array(self.GEN.A)
        self.populate_edge_weights()
    
    def set_star_topology(self):
        self.GEN.A = np.zeros([self.N+1,self.N+1])
        self.NXG1=nx.from_numpy_array(self.GEN.A)

    def set_sw_topology(self, p=.005,k=2):
        self.reset_topologies()
        self.GEN.watts_strogatz(p,k)
        self.NXG1=nx.from_numpy_array(self.GEN.A)
        self.populate_edge_weights()
    
    def set_sf_topology(self,num_seeds=5):
        self.reset_topologies()
        self.GEN.scale_free(num_seeds)
        self.NXG1=nx.from_numpy_array(self.GEN.A)
        self.populate_edge_weights()

    def set_custom_topology(self, A: np.ndarray):
        self.reset_topologies()
        self.A=A
        self.NXG1=nx.from_numpy_array(A)
        self.populate_edge_weights()

    def build_proximity_graph(self):
        self.A=np.zeros([self.N,self.N])
        for i in range(self.N):
            for j in range(i):
                if self.Euclidean(self.device_list[i],self.device_list[j]) < self.threshold:
                    self.A[i,j]=self.A[j,i]=1
        self.set_custom_topology(self.A)
        return
    
    def populate_edge_weights(self):  
        for (k,vals) in nx.to_dict_of_lists(self.NXG1).items():
            for v in vals:
                self.NXG1.add_edge(
                    k,v,weight=self.Euclidean(self.device_list[k], self.device_list[v])
                )
            # server_weight()
        # self.calculate_topology_cost() #TODO
        return
                
    def attribute_communities(self, memberships: dict=None):
        """ Membership Dictionary: is {node id: access_point (community_id)}
            Creates Community: {community_id: [list of members]}
        """
        if memberships != None: 
            self.ap_member_map=memberships
        self.communities = {ap: [] for ap in self.curr_apoints}
        
        for (k,AP) in self.ap_member_map.items():
            if (k!=AP):
                c=self.ap_color_map[AP] 
            else:
                c=self.ap_color
            
            self.NXG2.nodes[k]["hub"]=AP
            self.colors[c].append(k), self.communities[AP].append(k)
            
            if (k in self.curr_apoints) or (len(self.routes[k])<1):
                continue
            else:
                n=self.routes[k][1] # Add first neighbor ( assume routes are sorted)
                self.NXG2.add_edge(k,n, weight=self.NXG1.edges[(k,n)]["weight"])
        return
    
    def make_derived_network(self, memberships: dict=None):
        """ Taking Initial Input (NXG1) -- usually the proximity graph --
            create the derived graph which is either based on commmunities/ hub assignments etc.
            
            The derived graph also specifies the exact path of communication back to Hub. 
            This method should be extended by a particular algorithm.
        """
        if memberships != None: 
            self.ap_member_map=memberships
        
        for (AP,members) in self.ap_member_map.items():
            for k in members:
                self.NXG2.nodes[k]["hub"]=AP
                if (k in self.curr_apoints) or (len(self.routes[k])<1):
                    continue
                else:
                    n=self.routes[k][1] # Add first neighbor ( assume routes are sorted)
                    self.NXG2.add_edge(k,n, 
                        weight = self.Euclidean(self.device_list[k],self.device_list[n])
                    )
        return

    def find_closest_point(self,node,top=1,weight="weight"):
        """Iterate Over Hubs and Select the Best One"""
        G=self.NXG1 if (top == 1) else self.NXG2
        min_dist,best_route,assign=1e8,[],-1
        random.shuffle(self.curr_apoints)
        for ap in self.curr_apoints:
            try:
                route=nx.shortest_path(G,source=node,target=ap,weight=weight)
                if (len(route)<min_dist):
                    min_dist,best_route,assign=len(route),route,ap
            except nx.exception.NetworkXNoPath as E: # print(E)
                continue
        self.curr_apoints.sort()
        return assign, best_route

    def select_access_points_on_betweenness(self):
        """Betweenness Centrality Determines Hubs"""
        (access_points,_)=betweeness_rule(self.NXG1,top=self.num_apoints,weight='weight')
        self.curr_apoints=list(access_points)
        self.edge_nodes=list(set(self.NXG1.nodes)-set(access_points))
        self.generate_access_point_assignments()   
        self.reset_colors()     
        return

    def generate_access_point_assignments(self): 

        self.isolates=[]
        self.assignments["AP"]=self.curr_apoints
        ## assign devices
        for ap in self.curr_apoints:
            self.assignments[ap]={"access_point":ap, "route":[ap]} 
            self.ap_member_map[ap]=[ap]
            self.device_list[ap].parent_point = ap

        for node in self.edge_nodes:
            if node in self.curr_apoints:
                print(f"ERROR: AP should be excluded: {node}, {self.curr_apoints}")
                return
            
            (parent,route)=self.find_closest_point(node,weight="weight")
            self.routes[node]=route
            
            if (route == []):
                self.isolates.append(node)
                self.assignments[node]={"access_point":node, "route":[node]}
                self.device_list[node].parent_point = node
            else:
                self.assignments[node]={"access_point":parent, "route":route}
                self.assignments[parent]["route"].append(node)
                self.ap_member_map[parent].append(node)
                self.device_list[node].parent_point = parent
        
        if len(self.isolates) > 0:
            self.assignments["AP"].extend(self.isolates)
            for pt in self.isolates:
                if pt not in self.curr_apoints:
                    self.curr_apoints.append(pt)
        return

    # def generate_assignments(self, weight=None):## TODO: DEPRECATE

    #     """ TODO: DEPRECATE/LEGACY -- this is specifically structured for making yaml files
    #     """
        
    #     assignments,isolates,self.routes={},[],{}
    #     (access_points,_)=betweeness_rule(self.NXG1,top=self.num_apoints,weight=weight)
    #     self.curr_apoints=assignments["AP"]=list(access_points)
    #     self.reset_colors()
    #     peripherals=list(set(self.NXG1.nodes)-set(access_points))

    #     for ap in self.curr_apoints:
    #         assignments[ap]={"access_point":ap, "route":[ap]} 

    #     for n in peripherals:
    #         if n in self.curr_apoints:
    #             print("Error -- AP should be excluded:", n, self.curr_apoints)
    #             return
    #         (parent,route)=self.find_closest_point(n,weight=weight)
    #         self.routes[n]=route
    #         if (route == []):
    #             isolates.append(n)
    #             assignments[n]={"access_point":n, "route":[n]}
    #         else:
    #             assignments[n]={"access_point":parent, "route":route}
    #             assignments[parent]["route"].append(n)

    #     if len(isolates) > 0:
    #         assignments["AP"].extend(isolates)
 
    #         # self.ap_color_map={ap: clist[n] for (n,ap) in enumerate(self.curr_apoints)}

    #     self.assignments=assignments
    #     self.ap_member_map={k: v['access_point'] for (k,v) in assignments.items() if k != "AP"}
    #     self.reset_colors()
    #     # for (k,v) in self.routes.items():
    #     #     print(k, " ", v)

    #     return assignments

    def assignments_to_yaml(self, filename="sim_config.yaml"):
        with open(filename,"w") as f:
            yaml.dump(self.assignments,f,default_flow_style=False)
        return 

    def positions_to_yaml(self,path="node_coordinates.yaml"):
        with open(path,"w") as f:
            yaml.dump(
                {d.id: [d.x,d.y] for d in self.device_list},f,default_flow_style=False
            )
        return
    
    def plot_topology(self,top=1,edge_weights=False):
        self.plot_nodes(top)
        self.plot_edges(top,edge_weights)# NOTE: break up these steps for animation purposes
        return
    
    def plot_nodes(self,top=2):
        pos=self.get_positions()
        G,nodes=self.set_topology_to_plot(top)
        for node_color, nodelist in nodes.items():
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=node_color,node_size=250)
        labels = {x: x for x in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='w')

    def plot_edges(self,top=2,edge_weights=False):
        pos=self.get_positions()
        G,_=self.set_topology_to_plot(top)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges())
        if edge_weights == True:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            rounded_edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()} 
            # nx.draw_networkx_edge_labels(NXG, pos, edge_labels=rounded_edge_labels)
            for (k,v) in rounded_edge_labels.items():
                print(k, " : ", v)

    def set_topology_to_plot(self,top):
        if top == 1:
            NXG=self.NXG1        
            nodes={"lightseagreen": [dvc.id for dvc in self.device_list]}
        elif top == 2:
            NXG=self.NXG2
            nodes=self.colors
        else:
            print("error: invalid selection")
            return
        return (NXG,nodes)

    def plot_spring_communities(self):
        
        G=self.NXG2
        new_node_id = max(G.nodes) + 1
        G.add_node(new_node_id)
        for ap in self.curr_apoints:
            G.add_edge(new_node_id,ap)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, center=(0, 0),  k=0.5, iterations=150)
        for node_color, nodelist in self.colors.items():
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodelist, node_color=node_color, node_size=100
            )
        nx.draw_networkx_edges(G, pos, edgelist=G.edges())
        nx.draw_networkx_nodes(G, pos, nodelist=[new_node_id], 
                            node_color="gold", node_size=1200, node_shape='*', label="Server")
        return
    
    
    def plot_communities(self,ax=None, spring=False):

        G=self.NXG1
        pos=self.get_positions()
        
        if (spring==True):
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

        if ax==None:
            fig,ax=plt.subplots()

        for node_color, nodelist in self.colors.items():
            for node in nodelist:
                node_size,node_shape=230,'o'
                if node in self.curr_apoints:
                    node_size,node_shape=1000,'*'
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color,node_size=node_size,node_shape=node_shape,ax=ax)
            labels = {x: x for x in G.nodes}
            
        nx.draw_networkx_labels(G, pos, labels, font_size=14,font_weight='bold',font_color='w',ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(),width=.5, alpha=0.75,ax=ax)
    
    def plot_positions(self):
        pos=self.get_positions()
        NXG=self.NXG2
        fig=plt.figure()
        nodes={"lightseagreen": [dvc.id for dvc in self.device_list]}
        for node_color, nodelist in nodes.items():
            nx.draw_networkx_nodes(NXG, pos, nodelist=nodelist, node_color=node_color)
        return

    def calculate_topology_cost(self):
        self.topology_cost = 0
        for u, v, data in self.NXG2.edges(data=True):  #NOTE: for now this is simple distance squared
            self.topology_cost += (data['weight'])**2
        #TODO: also add the cost of aggregating each model here
        self.topology_cost += self.NXG2.number_of_edges()*self.AGG_COST
        print("Cost: ", self.topology_cost)
        return

    def move(self):
        """TODO: modify"""
        if (self.mobile == True):
            for d in self.device_list:
                d.λ = max(d.λ+np.random.normal(), 1.0)
                d.new_coordinate()
        return
    
    def calculate_distance_matrix(self):
        """TODO: delete"""
        self.D=np.zeros([self.n,self.n])
        for i in range(self.n):
            for j in range(i,self.n):
                self.D[i,j] = np.sqrt(
                    (self.devices[i].x-self.devices[j].x)**2 
                    + (self.devices[i].y-self.devices[j].y)**2
                )
        self.D = self.D + self.D.T
    
    def calculate_new_adjacency(self):
        """TODO: delete"""
        self.A = (self.D < self.d_max)*1
        for i in range(self.n):
            self.A[i,i]=0
    
    def get_positions(self):
        pos=[]
        for dvc in self.device_list:
            pos.append((dvc.x,dvc.y))
        return pos

    def ap_aggregate(self, results)-> List[torch.Tensor]:
        ap_params = {}
        for (AP, peers) in self.ap_member_map.items():
            chosen_params = [result[1] for result in results if result[0] in peers]
            self.ap_params[AP] = aggregate_params(chosen_params)

        return ap_params

## TODO:
# def cost_of_route(route,threshold=10):

#     cost = 0
#     tower_calls = 0
#     d2d_links=0
#     for (n,node) in enumerate(route):
#         if n == len(route)-1:
#             break
#         D=Euclidean(DEVCS[n],DEVCS[n+1])
#         if D > threshold:
#             c1 = (Euclidean(TOWER,DEVCS[n])**2)
#             c2 = (Euclidean(TOWER,DEVCS[n+1])**2)
#             cost += c1+c2
#             # print("Tower: ", c1+c2)
#             tower_calls += 1
#         else:
#             # print("D2D: ", (D**2))
#             cost += (D**2)
#             d2d_links+=1
        
#     print("Total Tower calls, ", tower_calls)
#     print("D2D links, ", d2d_links)
    
#     return cost

class GraphGenerator:
    """ 
        Build adjaceny matrix and save to a txt file.
        This aids in solving question 5 from homework 1.
    """
    def __init__(self,n=10):
        self.N=n
        self.M=0
        self.deg_pdf={}
        self.A = np.zeros([n,n], dtype=np.int16)
        for ii in range(n):
            self.deg_pdf[ii]=0.0

        return

    def random_pair(self):
        return( np.random.randint(self.N),  np.random.randint(self.N))

    def update_deg_dist(self):
        prob=lambda ii: self.get_degree(ii)/self.M/2

        for (k,v) in self.deg_pdf.items():
            self.deg_pdf[k]=prob(k)
            
        # print("check valid pdf: ", sum(self.deg_pdf.values()))

    def add_edge(self,i,j):
        if self.A[i,j] == 1: # Already added
            return
        if (i != j): # No Self Loops in these Models
            self.M += 1
            self.A[i,j] = self.A[j,i] = 1
        return
    
    def reset(self):
        self.A = np.zeros([self.N,self.N])
        self.M = 0
        return "Poof"
    
    def scale_free(self,num_seeds=5):
        # clear/reset adjacency
        self.reset()
        nodes=set([i for i in range(self.N)])
        seeds,sampler = [],[]
        
        # seed
        for _ in range(num_seeds):
            seed=random.sample(list(nodes),k=1)[0]
            nodes.remove(seed)
            seeds.append(seed)
        print(seeds)
        
        # randomly connect
        for _ in range(np.random.randint(12,24)): # num pulls=np.random.randint(12,24)
            (i,j)=random.sample(seeds,k=2)
            self.add_edge(i,j),sampler.append(i),sampler.append(j)
            # NOTE: i think using sampler mehthod is a bit simpler, I can check pdfs at the end (TODO)

        while (nodes):
            i=nodes.pop()
            candidates=random.sample(sampler,k=num_seeds) # connect to at most num_seeds
            for j in candidates:
                self.add_edge(i,j)
            
            if self.get_degree(i) < 1: #NOTE: ensure everybody gets a connection
                nodes.append(i)
        self.update_deg_dist()
    
    def erdos_renyi(self,p=0.05,num_edges=100, allow_isolates=True):
        self.M=0
        if (allow_isolates):
            while self.M < num_edges:
                self._random_edge(p=p)
        else:
            while (self._check_isolates()):
                self._random_edge(p=p)      
        self.update_deg_dist()

    def _random_edge(self, p=.05):
        (i,j)=self.random_pair()
        if (i==j) or (self.A[i,j] == 1):
            return
        if np.random.uniform() <= p:
            self.A[i,j] = self.A[j,i] = 1
            self.M += 1
    
    def _check_isolates(self):
        return np.any(np.sum(self.A,axis=1) == 0)
        
    def braid_lattice(self, step=2):
        self.reset()
        row=np.zeros(self.N)
        row[0:2*step+1]=1
        row[step]=0
        self.A=np.roll(
            np.stack([np.roll(row,k) for k in range(self.N)]), step,axis=0
        )
        self.M = np.sum(self.A)
    
    def remove_edge(self,i,j):
        self.M -= 1
        self.A[i,j] = self.A[j,i] = 0
        
    def watts_strogatz(self,p=.001,k=2):
        """
            p: rewiring probability
            k: steps in initial braid configuration (degree is actually 2k)
        """
        self.braid_lattice(step=k)
        nodes=[i for i in range(self.N)]

        # rewiring process
        for i in range(self.N):
            edges = list(np.nonzero(self.A[i,:])[0])
            complement = list(set(nodes)-set(edges)-set([i]))
            
            # NOTE: checking complement logic
            # for k in range(self.N):
            #     if k not in set(edges).union(complement):
            #         print(k)

            for e in edges:
                if np.random.rand() < p: 
                    new=random.sample(complement,1)[0]
                    # "edge" case, already connected then leave as is
                    if (self.A[i,new] == 1):
                        continue
                    else:
                        self.add_edge(i,new)
                        self.remove_edge(i,e)
                
        # double ensure edge count correct
        self.M=np.sum(self.A)//2
        return    

    def rank_nodes(self):
        return #TODO

    def degree_plot(self):
        return #TODO

    def get_degree(self,n):
        return np.sum(self.A[n,:])
    
    def write_file(self,fname="adjacency.txt"):
        np.savetxt(fname,self.A,fmt='%d')
    
    def to_adjacency_table(self):
        table={}
        for row in range(len(self.A)):
            neighbors = [int(n) for n in list(np.nonzero(self.A[row,:])[0])] #NOTE: annoying
            table[int(row)]=neighbors
        return table

    def edges_to_json(self,fname="edges.json"):
        table=self.to_adjacency_table()
        with open('er_sample.json', 'w') as fp:
            json.dump(table, fp)





##==================== NOTE: DEPRECATE ============================================================
# def rewire(self,adj_list: dict):
#     """ This is the original topology which we use to derive communities
#     """
#     self.reset_topologies()
#     G = nx.Graph()
#     for (k,vals) in adj_list.items():
#         devc=self.device_list[k]
#         # devc.x
#         for v in vals:
#             nbr=self.device_list[v]
#             G.add_edge(k,v, weight=self.Euclidean(
#                 self.device_list[k],
#                 self.device_list[v]
#             ))
#             #TODO: not actually sure if I need this
#             # nx.set_node_attributes(G,devc.x,"x")
#             # nx.set_node_attributes(G,devc.y,"y")
#     self.NXG1 = G # this is the original topology which we use to derive communities


# def dynamics():
#     Net=MobileNet(n=50)
#     steps=100

#     for step in range(steps):

#         # print("x: ", Net.devices[1].x)
#         # print("y: ", Net.devices[1].y)

#         Net.calculate_distance_matrix()
#         Net.calculate_new_adjacency()
#         # print(np.sum(Net.D))

#         # display the current configuration
#         Gx=nx.from_numpy_array(Net.A)
#         pos=Net.get_positions()
#         nx.draw(Gx,pos, with_labels=True, font_color="white")
#         plt.draw()
#         plt.pause(.15)
#         plt.cla()

#         # Update movement
#         for i in range(len(Net.devices.values())):
#             Net.devices[i].new_coordinate()