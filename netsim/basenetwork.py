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

## NOTE: (12/9) I think the basenetwork is actually kind of worthless now and I think everything can be incuded under basealgorithm

# NOTE: manual color map | use built-ins from matplotlib
COLORS = [
    "red", "blue", "gold", "green", "lavender", "magenta", "orange", "grey", "firebrick","teal",
    "tab:blue", "darkgreen", "brown", "indigo", "black", "bisque", "mediumturquoise", "darkviolet",
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


class Device:
    def __init__(self,id=0,x0=-97.7,y0=30.22):
        
        self.id=id
        self.parent_point=0  #NOTE: AP it is currently assigned to. Initial round before AP assignments, everyone is the same
        self.x,self.y = x0,y0    # BBOX = [-97.8395, -97.6819, 30.1961, 30.3511] #NOTE (from Mohawk)
        # self.color='red' #TODO :deprecate this

        # NOTE: New way stores path to param lookups instead of models themselves
        self.project_root=Path(__file__).resolve().parent.parent
        self.model_root=os.path.join(os.path.join(self.project_root,"devices"),"models")
        self.model_path=f"{self.model_root}/{id}.pth"
        self.community_model_path=f"{self.model_root}/{self.parent_point}.pth"
        self.coordinate_path=f"{self.project_root}/devices/movements/log_{self.id}.pkl"

    def assign_parent(self,apid):
        self.parent_point=apid
        self.commmunity_model_path=f"{self.model_root}/{self.parent_point}.pth"

    def get_model(self):
        return torch.load(self.model_path)
    
    def get_community_model(self):
        return torch.load(self.community_model_path)
    
    def set_step_coordinates(self,step=0): # Update positions at every GLOBAL round from files
        row=pd.read_pickle(self.coordinate_path).iloc[step]
        self.x,self.y=row['geolong'],row['geolat']
        return

# def Euclidean(d1:Device, d2:Device):
#         """NOTE: probably just rewrite this at the Network level to update distances every global round"""
#         return np.sqrt(
#             np.linalg.norm( np.array([d1.x,d1.y]) - np.array([d2.x,d2.y]))
#     )


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


class Network:
    def __init__(self,num_devices=10,num_classes=10,perceptual_map="nipy_spectral",threshold=10):
        
        self.N=num_devices # self.devices={}
        self.device_list=[Device(id=ii) for ii in range(self.N)]
        self.D = np.zeros([self.N,self.N]) 
        self.A = np.zeros([self.N,self.N])
        self.curr_apoints = []
        self.num_apoints = 5
        self.ap_color="black" # not really using this
        self.assignments = {} #NOTE: this might be legacy -- directly corresponds to the old file structure (AP, route: etc.)
        self.ap_member_map = {} #NOTE: key AP: values list of everyone assigned to it
        self.routes = {}
        self.ap_params = {}  #NOTE/TODO: store parameters in sim object, then devices can look up parameters

        # self.GEN=GraphGenerator(num_devices)
        self.threshold = threshold
        self.colormap = perceptual_map
        self.num_colors=self.num_apoints+1

        #NOTE: work on moving the plotting stuff to be totally independent
        self.cmap=get_uniform_colors(n=self.num_colors,colormap=perceptual_map) # TODO make even perceptual spacing dynamically at init
        self.reset_colors()
        self.reset_topologies()

        self.server=Device(id=101,x0=-97.6858,y0=30.2994) #NOTE server located at ATT tower on Manor Road
        self.TOTAL_COST=0
        self.server_round_costs=[] 
        self.communication_round_costs=[]

    def Euclidean(self,d1:Device, d2:Device): ## NOTE: Distance Cost Metric (Stored as Edge Weights on the Topology)
        return np.sqrt(
            np.linalg.norm( np.array([d1.x,d1.y]) - np.array([d2.x,d2.y]))
        )
    
    def server_distance(self, d1:Device):
        return np.sqrt(
            np.linalg.norm( np.array([d1.x,d1.y]) - np.array([self.server.x,self.server.y]))
        )

    def update_coordinates(self,step):
        for d in self.device_list:
            d.set_step_coordinates(step)

    ## NOTE: (New for Mielstone 2 | Cost Calculations)
    def calculate_server_distribution_aggregation_cost(self,mode="ap"): # NOTE, factor of 2 counts outbound/inbound 
        if mode == "star":
            return 2*np.sum([self.Euclidean(d,self.server) for d in self.device_list])
        elif mode == "ap":
            return 2*np.sum([self.Euclidean(self.device_list[ap],self.server) for ap in self.curr_apoints])
        else:
            print("unknown")


    def calculate_ap_distribution_aggregation_cost(self):
        return 2*np.sum(
            [np.sum([self.Euclidean(self.device_list[k],self.device_list[m]) for m in members]) for (k,members) in self.ap_member_map.items()]
        )
    
    def calculate_server_round_cost(self,star=False):
        if star:
            self.server_round_costs.append(self.calculate_server_distribution_aggregation_cost(mode="star"))
        else:
            self.server_round_costs.append(
                self.calculate_server_distribution_aggregation_cost(mode="ap") + self.calculate_ap_distribution_aggregation_cost()
            )
        
        self.TOTAL_COST+=self.server_round_costs[-1]
             

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

    def set_custom_topology(self, A: np.ndarray,which=1):
        """NOTE: I don't like this function. Mark to delete"""
        self.A=A
        G=nx.from_numpy_array(A)
        self.NXG1=self.populate_edge_weights(G)
    
    def generate_distances(self):
        coords = np.array([(device.x, device.y) for device in self.device_list])
        self.D = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)

    def fast_build_proximity_graph_(self, threshold=10):
        self.generate_distances()
        self.threshold = threshold
        self.A = (self.D < self.threshold).astype(int) #-identity
        np.fill_diagonal(self.A, 0)
        self.NXG1 = self.populate_edge_weights(G=nx.from_numpy_array(self.A))
        return

    def build_proximity_graph(self,threshold=10):
        """ NOTE: use version above ^^
        """
        self.A=np.zeros([self.N,self.N])
        self.threshold=threshold
        for i in range(self.N):
            for j in range(i):
                if self.Euclidean(self.device_list[i],self.device_list[j]) < self.threshold:
                    self.A[i,j]=self.A[j,i]=1
        self.NXG1=self.populate_edge_weights(G=nx.from_numpy_array(self.A))
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

    def select_access_points_on_betweenness(self, switch=False):
        """Betweenness Centrality Determines Hubs"""
        G = self.NXG2 if (switch) else self.NXG1
        (access_points,_)=betweeness_rule(G,top=self.num_apoints,weight='weight')
        self.curr_apoints=list(access_points)
        self.edge_nodes=list(set(G.nodes)-set(access_points))
        self.generate_access_point_assignments(switch=switch)
        self.reset_colors()     
        return

    def generate_access_point_assignments(self, switch=False):

        self.isolates=[]
        self.assignments["AP"]=self.curr_apoints
        top = 2 if (switch) else 1
        ## assign devices
        for ap in self.curr_apoints:
            self.assignments[ap]={"access_point":ap, "route":[ap]} 
            self.ap_member_map[ap]=[ap]
            self.device_list[ap].parent_point = ap

        for node in self.edge_nodes:
            if node in self.curr_apoints:
                print(f"ERROR: AP should be excluded: {node}, {self.curr_apoints}")
                return
            
            (parent,route)=self.find_closest_point(node,weight="weight", top=top)
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
