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
from pathlib import Path
import os

# NOTE: manual color map | use built-ins from matplotlib
COLORS = [
    "red", "blue", "gold", "green", "lavender", "magenta", "orange", "grey", "firebrick", "brown",
    "tab:blue", "darkgreen", "indigo", "black", "teal", "bisque", "mediumturquoise", "darkviolet",
    "aqua", "coral", "cyan", "hotpink", "lightgreen", "navy", "orchid", "slateblue", "tab:orange",
    "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "lightblue", "lime",
    "crimson", "maroon", "darkorange", "fuchsia", "turquoise", "salmon", "sienna", "tomato", "plum",
    "khaki", "peru", "violet", "steelblue", "darkkhaki", "skyblue", "thistle", "lightcoral", "rosybrown",
    "seagreen", "goldenrod", "slategrey", "cadetblue", "mediumvioletred", "darkcyan", "forestgreen",
    "mintcream", "papayawhip", "peachpuff", "powderblue", "purple", "royalblue", "saddlebrown", "yellow",
    "chartreuse", "dodgerblue", "deepskyblue", "dimgray", "gainsboro", "honeydew", "lightgoldenrodyellow",
    "lightgrey", "mistyrose", "moccasin", "navajowhite", "oldlace", "palegreen", "palevioletred", "seashell",
    "springgreen", "tan", "wheat", "whitesmoke"
]
ROOT=Path(__file__).resolve().parent.parent

# Utilities
def get_sampled_colors(n, colormap='viridis'):
    cmap = cm.get_cmap(colormap, n)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    random.shuffle(colors)
    return colors

def get_uniform_colors(n, colormap='viridis'):
    cmap = cm.get_cmap(colormap, n)
    colors = [cmap(i / (n - 1)) for i in range(n)]
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

class Device:
    def __init__(self,id=0,num_classes=10, x_max=100,y_max=100, λ=10,type=[4,5]):
        
        self.id=id
        self.parent_point=0 #NOTE: AP it is currently assigned to
        self.x=30.3
        self.y=71.7
        self.color='red' #TODO :deprecate this
        self.parent_point=0

        # NOTE: New way stores path to param lookups instead of models themselves
        self.project_root=Path(__file__).resolve().parent.parent
        self.model_root=os.path.join(os.path.join(self.project_root,"devices"),"models")
        self.model_path=f"{self.model_root}/{id}.pth"
        self.community_model_path=f"{self.model_root}/{self.parent_point}.pth"
        self.coordinate_path=f"{self.project_root}/devices/movements/log_{self.id}.pkl"

        # Legacy Stuff
        # NOTE: Legacy -- randomly assigned initial positions, instead now we read from device_movements/
        # self.x=np.random.randint(-y_max,y_max)
        # self.y=np.random.randint(-x_max,x_max)

        # NOTE: DELETE these once switching over to files is done
        # self.model=Net(num_classes)
        # self.community_model=Net(num_classes)  #NOTE: really these are community parameters, after a global round they are the global params

    def assign_parent(self,apid):
        self.parent_point=apid
        self.commmunity_model_path=f"{self.model_root}/{self.parent_point}.pth"

    def get_model(self):
        return torch.load(self.model_path)
    
    def get_community_model(self):
        return torch.load(self.community_model_path)
    
    def set_step_coordinates(self,step=0):
        #NOTE: update positions at every GLOBAL round
        row=pd.read_pickle(self.coordinate_path).iloc[step]
        self.x,self.y=row['geolat'],row['geolong']
        return
    
#NOTE: Device Level Calcualtions
# def compare_device_cosines_new(d1: Device, d2: Device): #(Pass Device Objects)
#     """TODO: test with training loop -- 
#     """
#     state_dict_1=d1.get_model()
#     state_dict_2=d2.get_community_model()
#     #return compute_cosine_sim(state_dict_1,state_dict_2)
#     print(state_dict_2.keys())
#     print(state_dict_1.keys())
#     return 0

def Euclidean(d1:Device, d2:Device):
        """NOTE: probably just rewrite this at the Network level to update distances every global round"""
        return np.sqrt(
            np.linalg.norm( np.array([d1.x,d1.y]) - np.array([d2.x,d2.y]))
    )

class Network:
    def __init__(self,num_devices=10,num_classes=10,perceptual_map="nipy_spectral",threshold=10):
        
        self.N=num_devices # self.devices={}
        self.device_list=[Device(i,num_classes=num_classes) for i in range(self.N)]
        self.D = np.zeros([self.N,self.N]) #NOTE: not in use
        self.A = np.zeros([self.N,self.N])
        self.curr_apoints = []
        self.num_apoints = 5
        self.ap_color="black" # not really using this
        self.assignments = {} #NOTE: this might be legacy -- directly corresponds to the old file structure (AP, route: etc.)
        self.ap_member_map = {} #NOTE: key AP: values list of everyone assigned to it
        self.routes = {}
        self.ap_params = {}  #NOTE/TODO: store parameters in sim object, then devices can look up parameters

        self.GEN=GraphGenerator(num_devices)
        self.threshold = threshold
        self.colormap = perceptual_map
        self.num_colors=self.num_apoints+1

        #NOTE: work on moving the plotting stuff to be totally independent
        self.cmap=get_uniform_colors(n=self.num_colors,colormap=perceptual_map) # TODO make even perceptual spacing dynamically at init
        self.reset_colors()
        self.reset_topologies()

        #NOTE: choose location of server or somehow otherwise compute the cost of talking to server
        self.server=Device()
        self.server.x,self.server.y = 30.2994,-97.6858 # ATT tower on Manor Road
        self.total_cost = 0

    def Euclidean(self,d1:Device, d2:Device):
        return np.sqrt(
            np.linalg.norm( np.array([d1.x,d1.y]) - np.array([d2.x,d2.y]))
        )

    def update_coordinates(self,step):
        """NOTE: the 'movement' function from data
        """
        for d in self.device_list:
            d.set_step_coordinates(step)

    def reset_topologies(self):
        self.A = np.zeros([self.N,self.N])
        self.NXG1 = nx.from_numpy_array(self.A) # sparse communities
        self.NXG2 = nx.from_numpy_array(self.A)
        self.topology_cost = 0

    def reset_colors(self, ap_color=None):
        # NOTE/TODO: Possibly Change this to do all the plotting after the fact
        """
            Assign/ populate colors from the ap_member_map and matpltlib color_map
        """
        self.colors={}
        for (n,(ap,vals)) in enumerate(sorted(self.ap_member_map.items())):
            if len(self.cmap) > n:
                #print("n ", n, "length ", len(self.cmap))
                self.colors[self.cmap[n]]=vals
            else:
                ii=n-len(self.cmap)
                self.colors[COLORS[ii]] = vals # extra colors for loners
        if ap_color is not None:
            self.colors[ap_color]=self.curr_apoints
        
    def init_server(self,pos=[0,0]):
        self.server = Device()
        self.server.x = pos[0]
        self.server.y = pos[1]
        return
    
    ##TODO: Fix this Redundancy
    def populate_edge_weights(self,G):
        for (k,vals) in nx.to_dict_of_lists(G).items():
            for v in vals:
                G.add_edge(
                    k,v,weight=self.Euclidean(self.device_list[k], self.device_list[v])
                )
        return G

    ## NOTE: Creating Baseline Topologies (TODO: possibly move these to a Diffferent Level)
    def set_er_topology(self, p=.05,num_edges=100,allow_isolates=False):
        self.reset_topologies()
        self.GEN.erdos_renyi(p,num_edges,allow_isolates)
        G=nx.from_numpy_array(self.GEN.A)
        self.NXG1=self.populate_edge_weights(G)
    
    def set_star_topology(self):
        self.GEN.A = np.zeros([self.N+1,self.N+1])
        self.NXG1=nx.from_numpy_array(self.GEN.A)

    def set_sw_topology(self, p=.005,k=2):
        self.reset_topologies()
        self.GEN.watts_strogatz(p,k)
        G=nx.from_numpy_array(self.GEN.A)
        self.NXG1=self.populate_edge_weights(G)
    
    def set_sf_topology(self,num_seeds=5):
        self.reset_topologies()
        self.GEN.scale_free(num_seeds)
        G=nx.from_numpy_array(self.GEN.A)
        self.NXG1=self.populate_edge_weights(G)

    def set_custom_topology(self, A: np.ndarray,which=1):
        """NOTE: I don't like this function. Mark to delete
        
        """
        # self.reset_topologies()
        self.A=A
        G=nx.from_numpy_array(A)
        self.NXG1=self.populate_edge_weights(G)

    def fast_build_proximity_graph_(self, threshold=10):
        self.threshold = threshold
        coords = np.array([(device.x, device.y) for device in self.device_list])
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        self.A = (distances < self.threshold).astype(int)
        np.fill_diagonal(self.A, 0)
        self.NXG1 = self.populate_edge_weights(G=nx.from_numpy_array(self.A))
        return

    def build_proximity_graph(self,threshold=10):
        self.A=np.zeros([self.N,self.N])
        self.threshold=threshold
        for i in range(self.N):
            for j in range(i):
                if self.Euclidean(self.device_list[i],self.device_list[j]) < self.threshold:
                    self.A[i,j]=self.A[j,i]=1
        self.NXG1=self.populate_edge_weights(G=nx.from_numpy_array(self.A))
        # print(self.NXG1.edges())
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
        """ NOTE:
            Encode the two-step proceess where NXG1 is assigned links (or decentralized links)
            NXG2 stores the actual routes of device aggregation. 
            In  milestone 1, the first topology was used to assign the betweeness values,
            then NXG2 was derived from this where each node is explicitly assigned to an AP
        
        
            Taking Initial Input (NXG1) -- usually the proximity graph --
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

    ## NOTE: Betweenness Logic
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
    
    def set_topology_to_plot(self,which=1):
        if which == 1:
            return (self.NXG1,{"blue":self.NXG1.nodes()})
        if which == 2:
            return (self.NXG2,{"teal":self.NXG2.nodes()})

    
    #NOTE: Plotting Methods
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

    def plot_communities(self,ax=None, spring=False,which=2):

        G=self.NXG2 if which == 2 else self.NXG1
        pos=self.get_positions()
        
        if (spring==True):
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

        if ax==None:
            fig,ax=plt.subplots()

        for node_color, nodelist in self.colors.items():
            for node in nodelist:
                node_size,node_shape,fsize,fweight,alpha,fontcolor=180,'o',9,"normal",0.5,"black"
                if node in self.curr_apoints:
                    node_size,node_shape,fsize,fweight,alpha,fontcolor=1700,'*',13,"bold",1.0,"white"
                nx.draw_networkx_nodes(G, pos, nodelist=[node],
                    node_color=node_color,node_size=node_size,node_shape=node_shape,alpha=alpha,ax=ax)
                nx.draw_networkx_labels(G, pos, {node:node}, font_size=fsize,font_weight=fweight,font_color=fontcolor,ax=ax)
            # labels = {x: x for x in G.nodes}
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(),width=.35, alpha=0.55,ax=ax)

    ## NOTE: New for Mielstone 2
    def calculate_server_distribution_aggregation_cost(self,mode="ap"):
        SERVER_LOC = [30.2994,-97.6858]  # ATT Server
        if mode == "star":
            for d in self.device_list:
                self.total_cost+=self.Euclidean(d,self.server)
        elif mode == "ap":
            for ap in self.curr_apoints:
                self.total_cost+=self.Euclidean(self.device_list[ap],self.server)
        else:
            print("unknown")

    def calculate_ap_distribution_aggregation_cost(self):
        for (k,members) in self.ap_member_map.items():
            self.total_cost+=np.sum([self.Euclidean(self.device_list[k],self.device_list[m]) for m in members])










# TODO : Deprectate this (??)
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