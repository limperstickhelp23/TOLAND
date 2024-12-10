import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from .basenetwork import *

### NOTE: Calculation Utilities Ubiquitous to Multiple Algorithms
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

def batch_cosine_similarity(device_list):
    """ batch-wise calculation of cosine similarities.
    """
    model_matrix = torch.stack([torch.cat([param.flatten() for param in device.get_model().values()]) for device in device_list]) # Stack model parameters and Normalize
    model_matrix = model_matrix / model_matrix.norm(dim=1, keepdim=True)
    return model_matrix @ model_matrix.T

def device_neighbor_similarity(model_params, neighbors, device_list):
    """ Calculates a device's model to each of its neighboring node's community model
            :param model_params:
            :param neighbors:
            :param model_params:
            :return: 1D tensor
    """
    # Flatten and normalize the given model's parameters
    d = "mps" if torch.backends.mps.is_available() else "cpu"
    target_model_vector = torch.cat([param.flatten() for param in model_params.values()]).to(device=d)
    target_model_vector /= torch.norm(target_model_vector)

    # Prepare and normalize device models in a batch
    device_model_matrix = torch.stack([
        torch.cat([param.flatten() for param in device_list[nid].get_community_model().values()])
        for nid in neighbors
    ]).to(device=d)
    device_model_matrix /= device_model_matrix.norm(dim=1, keepdim=True)
    cosine_similarities = torch.matmul(device_model_matrix, target_model_vector)

    return {nid: similarity.item() for nid, similarity in zip(neighbors, cosine_similarities)} ## Similarity Dict.

#NOTE NEW Co-Sim Functions
def compute_cosim_metric_from_state(state_dict1: dict, state_dict2: dict):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    tensor_1 = torch.cat([param.flatten() for param in state_dict1.values()])
    tensor_2 = torch.cat([param.flatten() for param in state_dict2.values()])

    tensor_1 = tensor_1 / tensor_1.norm()
    tensor_2 = tensor_2 / tensor_2.norm()

    cosine_sim_matrix = cos(tensor_1, tensor_2)
    return cosine_sim_matrix

def compute_model_to_community_cosim(dvc1: Device, dvc2: Device):
    return compute_cosim_metric_from_state(dvc1.get_model(),dvc2.get_community_model())

def compute_model_to_model_cosim(dvc1: Device, dvc2: Device):
    return compute_cosim_metric_from_state(dvc1.get_model(),dvc2.get_model())

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

######### Utilities ###########
def get_sampled_colors(n, colormap='viridis'):
    ## TODO : Check are we using ??
    cmap = cm.get_cmap(colormap, n)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    random.shuffle(colors)
    return colors

def get_uniform_colors(n, colormap='viridis'): ## TODO using it ??? 
    cmap = cm.get_cmap(colormap, n)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    return colors

def sort_dictionary(d,desc=True):
    return sorted(d.items(), key = lambda item: item[1], reverse=desc)




class Algorithm(Network):
    """ 
        It is just another wrapper for a few abstract methdods hanlding community assignment and local aggregating.
    """
    
    def __init__(self,num_devices=25,num_classes=10,perceptual_map="turbo",threshold=10,star=False):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map,threshold=threshold)
        self.ap_param_stacks={}
        self.sf_seeds=max(2,int((.10)*num_devices))
        self.update_coordinates(0)
        self.star = star

    def reset_cost_collection(self):
        self.community_total_edge_cost=0  # Cost of Intra-Community Communication
        self.server_communication_cost=0  # Cost of Communication to/from Server for the given access points

    def run_global_round_setup_steps(self,round=0, threshold=10): #NOTE: Generally contains these 2 steps
        
        self.update_coordinates(round) # If movement enabled -- get coordinates from device files
        self.run_linking_algorithm() # Custom Re-linking methods
        
        if (self.star): #NOTE The trivial map which includes all
            N=len(self.device_list)
            self.ap_member_map={N:[i for i in range(N)]} 
        return None
    
    def run_local_aggregation_round(self):        
        """ The local aggregation round maps parameters internally.
            This way each device knows its community parameters.
            The Trivial star map is just a single entry with all devices.
        """
        ap_avg_state_dict={} #NOTE: temp variable to match back semantically to cloud.py
        for (AP,members) in self.ap_member_map.items():
            #NOTE New Step: Load from files using the current ap_member_map
            ap_avg_state_dict[AP]=aggregate_params(
                [torch.load(self.device_list[member].model_path) for member in members]
            )
        #NOTE For additional logic, override in child class
        return ap_avg_state_dict
    
    def run_linking_algorithm(self,star=False):
        """Just a default behavior for place-holding. Pass "star" Flag to do nothing at this round (trivial behavior)"""
        if (self.star): return
        self.build_proximity_graph(threshold=10)
        self.select_access_points_on_betweenness()

    def map_results_to_files(self,results):
        for (n,model,device) in results:

            torch.save(model, self.device_list[n].model_path)
    
    def assign_communities(self):
        """ 
            Reset Communities After Organizing & After Training
                - ap_member_map: Key(AP) Value(List of Members)
                - ap_param_stacks: Directly Looks Up From Each Device Model
        """
        self.ap_member_map={ap:[] for ap in self.curr_apoints}
        for d in self.device_list:
            self.ap_member_map[d.parent_point].append(d.id)
        self.reset_colors()




# #NOTE: Methods Ubiquitous to Varios Algortihms
# def compute_cosim_metric(m1: nn.Parameter, m2: nn.Parameter):
#     sim,cos=0, nn.CosineSimilarity(dim=0, eps=1e-6)
#     for (p1,p2) in zip(m1,m2):
#         layer_sim=abs(cos(p1.data.flatten(),p2.data.flatten()).item())
#         sim += layer_sim
#     return sim

# def compute_device_model_cosim_metric(dvc1, dvc2):
#     m1=dvc1.model.parameters()
#     m2=dvc2.model.parameters()
#     return compute_cosim_metric(m1,m2)

# def compute_device_to_community_cosim(dvc1, dvc2):
#     m1=dvc1.model.parameters()
#     m2=dvc2.community_model.parameters()
#     return compute_cosim_metric(m1,m2)

# def batch_cosine_similarity(device_list):
#     """
#     batch wise calculation of cosine similarities.
#     #TODO/NOTE:
#     """
#     # Stack model parameters and Normalize
#     model_matrix = torch.stack([torch.cat([param.flatten() for param in device.get_model().values()]) for device in device_list])
#     model_matrix = model_matrix / model_matrix.norm(dim=1, keepdim=True)

#     cosine_sim_matrix = model_matrix @ model_matrix.T

#     return cosine_sim_matrix

# def device_neighbor_similarity(model_params, neighbors, device_list):
#     """
#     Calculates a device's model to each of its neighboring node's community model
#     :param model_params:
#     :param neighbors:
#     :param model_params:
#     :return: 1D tenosr
#     """
#     # Flatten and normalize the given model's parameters
#     d = "mps" if torch.backends.mps.is_available() else "cpu"
#     target_model_vector = torch.cat([param.flatten() for param in model_params.values()]).to(device=d)
#     target_model_vector /= torch.norm(target_model_vector)

#     # Prepare and normalize device models in a batch
#     device_model_matrix = torch.stack([
#         torch.cat([param.flatten() for param in device_list[nid].get_community_model().values()])
#         for nid in neighbors
#     ]).to(device=d)
#     device_model_matrix /= device_model_matrix.norm(dim=1, keepdim=True)

#     cosine_similarities = torch.matmul(device_model_matrix, target_model_vector)

#     similarity_dict = {nid: similarity.item() for nid, similarity in zip(neighbors, cosine_similarities)}

#     return similarity_dict

# #NOTE NEW Co-Sim Functions
# def compute_cosim_metric_from_state(state_dict1: dict, state_dict2: dict):
#     cos = nn.CosineSimilarity(dim=0, eps=1e-6)
#     tensor_1 = torch.cat([param.flatten() for param in state_dict1.values()])
#     tensor_2 = torch.cat([param.flatten() for param in state_dict2.values()])

#     tensor_1 = tensor_1 / tensor_1.norm()
#     tensor_2 = tensor_2 / tensor_2.norm()

#     cosine_sim_matrix = cos(tensor_1, tensor_2)
#     return cosine_sim_matrix

# def compute_model_to_community_cosim(dvc1: Device, dvc2: Device):
#     return compute_cosim_metric_from_state(dvc1.get_model(),dvc2.get_community_model())

# def compute_model_to_model_cosim(dvc1: Device, dvc2: Device):
#     return compute_cosim_metric_from_state(dvc1.get_model(),dvc2.get_model())

# def betweeness_rule(G: nx.Graph, top = 10, weight=None):
#     return zip(*sort_dictionary(nx.betweenness_centrality(G,weight=weight))[:top])

# def aggregate_params(model_params):
#     averaged_state_dict = {}

#     for key in model_params[0].keys(): #conv1_
#         param_stack = torch.stack([state_dict[key] for state_dict in model_params], dim=0)
#         avg_params = torch.mean(param_stack, dim=0)
#         averaged_state_dict[key] = avg_params

#     return averaged_state_dict
