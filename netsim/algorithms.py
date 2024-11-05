from netsim.network import *
import os,random
import numpy as np
import networkx as nx
import torch.nn as nn

#NOTE: Some UTILS
def compute_cosim_metric(m1: nn.Parameter, m2: nn.Parameter):
    #TODO/NOTE: this has to change if we move to using files instead
    sim,cos=0, nn.CosineSimilarity(dim=0, eps=1e-6)
    for (p1,p2) in zip(m1,m2):
        layer_sim=abs(cos(p1.data.flatten(),p2.data.flatten()).item())
        sim += layer_sim
    return sim

def compute_device_model_cosim_metric(dvc1, dvc2):
    #TODO/NOTE: this has to change if we move to using files instead
    m1=dvc1.model.parameters()
    m2=dvc2.model.parameters()
    return compute_cosim_metric(m1,m2)

def compute_device_to_community_cosim(dvc1, dvc2):
    #TODO/NOTE: this has to change if we move to using files instead
    m1=dvc1.model.parameters()
    m2=dvc2.community_model.parameters()
    return compute_cosim_metric(m1,m2)

#NOTE NEW Co-Sim Functions
def compute_cosim_metric_from_state(state_dict1: dict, state_dict2: dict):
    sim = 0
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    # Iterate through the keys in the state dictionaries
    for key in state_dict1:
        if key in state_dict2:    # Flatten the tensors and compute cosine similarity
            sim += abs(cos(state_dict1[key].flatten(), state_dict2[key].flatten()).item())
    return sim

def compute_model_to_community_cosim(dvc1: Device, dvc2: Device):
    return compute_cosim_metric_from_state(dvc1.get_model(),dvc2.get_community_model())

def compute_model_to_model_cosim(dvc1: Device, dvc2: Device):
    return compute_cosim_metric_from_state(dvc1.get_model(),dvc2.get_model())


class Algorithm(Network):
    """ 
        Honestly... this could probably be added to Mobile Net Class.
        It is just another wrapper for a few abstract methdods hanlding community assignment and local aggregating.
    """
    
    def __init__(self,num_devices=25,num_classes=10,perceptual_map="turbo",threshold=10):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map,threshold=threshold)
        self.ap_param_stacks={}
        self.sf_seeds=max(2,int((.10)*num_devices))
    
    def run_global_round_setup_steps(self,round=0):
        print("Implement in child classes")
        
        #NOTE: i think generally just contains these 2 steps
        
        # Simulate Movements
        self.update_coordinates(round) # If movement enabled
        
        # Re-linking Algorithm
        self.run_linking_algorithm() # Custom relinking methods
        
        # whatever else TODO before training 
      
        return None
    
    def run_local_aggregation_round(self):        
        """ The local aggregation round maps parameters internally.
                This way each device knows its community parameters.
        """
        ap_avg_state_dict={} #NOTE: temp variable to match back semantically to cloud.py
        for (AP,members) in self.ap_member_map.items():
            #NOTE New Step: First load param_stacks based on files and current ap_member_map
            # ap_parameter_stack=[torch.load(self.device_list[member].model_path) for member in members]
            ap_avg_state_dict[AP]=aggregate_params(
                [torch.load(self.device_list[member].model_path) for member in members]
            )
        #NOTE For additional logic, override in child class
        return ap_avg_state_dict
    
    def run_linking_algorithm(self):
        #TODO: just a default -- in general this should be unique to each algorithm 
        self.build_proximity_graph(threshold=10)
        self.select_access_points_on_betweenness()
        print(self.ap_member_map)

    def map_results_to_files(self,results):
        for (n,model,device) in results:
            # print(type(model), " " , len(model), " ", model[0], " ",model[2])
            torch.save(model, self.device_list[n].model_path)
    
    def assign_communities(self):
        """
        Simulated Version: (Same as Integrated Version ?)
            Re-Set Communities After Organizing & After Training 
                ap_member_map: Key(AP) Value(List of Members)
                ap_param_stacks: Directly Looks Up From Each Device Model
        """
        self.ap_member_map={ap:[] for ap in self.curr_apoints}

        for d in self.device_list:
            self.ap_member_map[d.parent_point].append(d.id)
            # self.ap_param_stacks[d.parent_point].append(d.model.state_dict()) #NOTE: DEPRECATE not needed anymore
        
        self.reset_colors() # NOTE: whoops forgot this before

#BASELINE
class Baselines(Algorithm):
    """ Set Baseline Topologies
    """
    def __init__(self,num_devices=25,num_classes=10,perceptual_map="turbo",topology="star",threshold=10):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map,threshold=threshold)
        self.dynamic=False
        self.topology=topology

        if topology == "star":
            self.set_star_topology()
        else:
            self.init_topology()
    
    def rewire_round(self):
        if self.dynamic==False:
            print("static: do nothing")
        else:
            print("dynamic: new topology")
            self.init_topology()
        return

    def init_topology(self):
        if self.topology == "scalefree":
                self.set_sf_topology()
        elif self.topology == "random":
            self.set_er_topology()
        else:
            self.set_er_topology()
        
        print(f"Init: {self.topology}")
        self.select_access_points_on_betweenness()
        self.make_derived_network()
        self.assign_communities()

#ALGORITHMS
class LocalizedPreferentialAttachment(Algorithm):
    """ TODO:
        Set Hubs Based on Preferential Attachment mechanism but where it is also based on location at every round. 
        
        Essentially: Scale Free + Proximity

        NOTE: 
        This could be used as a first step and cosine reassignemnt could be used as a second. Or it could do well on its own. 
        Compare IID and Non-IID cases.
    """
    def __init__():
        ##TODO
        return

class ScaleFreeRewiring(Algorithm): 
    """ The Difference here from the baseline is that each round is a new scale-free topology. 
    Comparable to Star ? What about if we re-use the same ScaleFree Topology ?
    """
    
    def __init__(self,num_devices=25,num_classes=10,perceptual_map="turbo",threshold=10):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map,threshold=threshold)
        ratio=.07
        self.sf_seeds=max(int(ratio*num_devices),2)

    def run_global_round_setup_steps(self, round=0):
        
        self.update_coordinates(round)
        self.run_linking_algorithm(round_num=round)
        return 

    def run_linking_algorithm(self,round_num=0,folder="static_updates/",save=False,weight="weight"):
        """ Cases:
                - Static Devices/Static Topology
                - Static Device/Changing Topology
                - Mobile Devices/Changing Topology
        """
        self.set_sf_topology(num_seeds=self.sf_seeds) #NOTE: randomly generates new Scale Free Graph
        self.select_access_points_on_betweenness()
        self.make_derived_network()
        self.assign_communities()
        # if (save): #TODO: yaml stuff still on hold
        #     self.assignments_to_yaml(os.path.join(folder,f"round_{round_num}.yaml"))

class CosineReassignment(Algorithm):
    """
        Goal: Try to reassign devices to other communities based on cosine similarity
        but try to do so while remaining cost-efficient. 
        
        Naive Idea:
            - First Round everyone trains and the community models are aggregated
            - Then everyone gets back the "community model"
            - Next, compare own local model to each of their neighbors community models
            - Then they reassign themselves to the community with most similarity
            - Then they broadcast their desired community to devices around them
            
        TODO: ReLinking/ Rerouting Steps
            - Any common communities will form a link from proximity neighbors and if none then they will remain isolated
            - The last problem to solve now is flow of information / Select Aggregation Route for Lowest Cost (Decentralized Manner?)

    """

    def __init__(self,num_devices,num_classes,perceptual_map,threshold):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map,threshold=threshold)
        self.ap_param_stacks={}

    def run_local_aggregation_round(self,max_iterations=3):
        self.compare_communities(max_iters=max_iterations)  
        return super().run_local_aggregation_round()
    
    def run_global_round_setup_steps(self, round=0, threshold=10):
        self.update_coordinates(round)
        self.run_linking_algorithm(threshold)
        return 

    def run_linking_algorithm(self,threshold=10):
        self.threshold=threshold
        self.build_proximity_graph() # allow_isolates = False
        self.select_access_points_on_betweenness() 
        self.make_derived_network()
        self.assign_communities()
        return

    def compare_communities(self,max_iters=10):
        """
            Devices store parameters from both their own model and the community model.
            Then compare their model to the community models of their neighbors to decide if the neighbor's 
            community is a better fit. Also compare to their own current community model. If no community is better then no switch occurs
        """
        changes=1000

        for iter in range(max_iters):
            if changes < 2:
                break
            else:
                changes = 0

            for dobj in self.device_list:
                
                dvc=dobj.id
                if dvc in self.curr_apoints: #APoints stay fixed
                    continue
                
                ## Similarity with "self" -- community model
                max_cosim=compute_model_to_community_cosim(self.device_list[dvc],self.device_list[dvc])
                max_nbr,max_color=dvc,self.device_list[dvc].color
            
                for nbr in self.NXG1.neighbors(dvc):
                    
                    # Similarity with it's neighbor's community
                    cosim=compute_model_to_community_cosim(self.device_list[dvc], self.device_list[nbr])
                    if cosim > max_cosim:
                        max_cosim,max_nbr,max_color=cosim,nbr,self.device_list[nbr].color
                        self.device_list[dvc].parent_point = self.device_list[nbr].parent_point
                
                # print(f"match? {dvc} : {max_nbr}, {max_cosim}")
                if (max_nbr != dvc) and (iter == 0):
                    changes += 1
                    print(f"Max Influence on {dvc}: {max_nbr}, {max_color}")
                    ## print(self.device_list[dvc].community_model.state_dict()['conv1.bias'])
                    ## print(self.device_list[nbr].community_model.state_dict()['conv1.bias'])
                
                self.device_list[dvc].color = max_color
            
        # Re-Set Communities after Comparisons
        self.assign_communities()
        return
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
###NOTE: Other Ideas | Can Ignore | Prototype New Ideas Etc. 
class EfficientLessCentralizedServer(Algorithm):
    def __init__(self,num_devices=20):
        super().__init__(n=num_devices) 
        self.mobile=False   
        self.ap_color="black"
    
    def rewire_round(self,round_num=0):
        indices=[i for i in range(self.N)]
        clusters=random.sample([i for i in range(self.N)], self.num_apoints)
        indices=list(set(indices)-set(clusters))
        self.curr_apoints=list(clusters)
        memberships={}

        for dvc in indices:
            head=random.sample(clusters,1)[0]
            memberships[dvc]=head
            self.routes[dvc]=[dvc,head]
            self.A[dvc,head]=self.A[head,dvc]=1
        
        self.set_custom_topology(self.A)
        self.reset_colors()
        self.attribute_communities(memberships)

class MinSpanTreeServer(Algorithm):
    def __init__(self,num_devices=20,λ=10):
        super().__init__(n=num_devices,λ=λ)
        self.mobile=True
        self.weight="weight"
        self.save=False
        self.outdir="mst_tree_updates/"
        self.method="fully_connected"
    
    def rewire_round(self,round_num=0):

        if self.method == "fully_connected":
            A=np.ones([self.N,self.N])
            np.fill_diagonal(A, 0)
            self.set_custom_topology(A)
        elif self.method == "proximity":
            self.build_proximity_graph()
            self.set_custom_topology(self.A)
        else:
            print("method not recognized")
            return
    
        self.set_custom_topology(
            nx.adjacency_matrix(
                nx.minimum_spanning_tree(self.NXG1,weight=self.weight,algorithm="kruskal")
            )
        )
        # self.generate_assignments(weight=self.weight)
        self.attribute_communities()
        # if (save):
        #     self.assignments_to_yaml(os.path.join(folder,f"round_{round_num}.yaml"))

class MaxEntropyServer(Algorithm):
    def __init__(self,num_devices=20):
        super().__init__(n=num_devices)   

## NOTE: Add New Algorithms here
class MyNewAlgorithm(Algorithm):
    def __init__(self,num_devices=20):
        super().__init__(n=num_devices)   
    
    def rewire_round(self):
        return




#   def self_assign(self,max_iters=10):
#         """
#         NOTE: FIRST VERSION  -- New version usese comparisons to community models instad.
       
#         This works ok except sometimes the order can affect it. 
#         So for example (0,3,6,9) are all supposed to be red but 9 is initially yellow.
#         Then 3 was similar to 9 so was pulled into yellow camp and brought 6 with it. 

#         This is at the first round. To get around this, I hope doing community aggregations will help.

#         Another way would be to try to assign them to the most similar access point. However they may not always be close
#         to the access point with the best similarity
#         """

#         changes=1000

#         for _ in range(max_iters):
#             if changes < 2:
#                 break
#             else:
#                 changes = 0
            

#             for dobj in self.device_list:
                
#                 dvc=dobj.id
#                 if dvc in self.curr_apoints:
#                     continue # NOTE: access points stay fixed, I think this will be key to algorithm but could be wrong
                
#                 ## Similarity with "self" -- community model
#                 max_cosim=compute_device_to_community_cosim(self.device_list[dvc],self.device_list[dvc])
#                 max_nbr,max_color=dvc,self.device_list[dvc].color

#                 for nbr in self.NXG1.neighbors(dvc):

#                     cosim=compute_device_model_cosim_metric(self.device_list[dvc], self.device_list[nbr])
#                     if cosim > max_cosim:
#                         max_cosim,max_nbr,max_color=cosim,nbr,self.device_list[nbr].color
#                         self.device_list[dvc].parent_point = self.device_list[nbr].parent_point

#                 # print(f"match? {dvc} : {max_nbr}, {max_cosim}")
#                 if max_color != self.device_list[dvc].color:
#                     changes += 1
                
#                 self.device_list[dvc].color = max_color   
#         # TODO/NOTE: load current access points from devices -- shoul dbe its own function  
#         # Re-Set Communities after Organization
#         self.assign_communities()
#         return