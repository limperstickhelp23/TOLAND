from netsim.network import *
import os,random
import numpy as np
import networkx as nx
import torch.nn as nn
import random
import copy
from tqdm import tqdm

#NOTE: Some UTILS
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
    """
    batch wise calculation of cosine similarities.
    #TODO/NOTE:
    """
    # Stack model parameters and Normalize
    model_matrix = torch.stack([torch.cat([param.flatten() for param in device.get_model().values()]) for device in device_list])
    model_matrix = model_matrix / model_matrix.norm(dim=1, keepdim=True)

    cosine_sim_matrix = model_matrix @ model_matrix.T

    return cosine_sim_matrix

def device_neighbor_similarity(model_params, neighbors, device_list):
    """
    Calculates a device's model to each of its neighboring node's model
    :param model_params:
    :param neighbors:
    :param model_params:
    :return: 1D tenosr
    """
    # Flatten and normalize the given model's parameters
    d = "mps" if torch.backends.mps.is_available() else "cpu"
    target_model_vector = torch.cat([param.flatten() for param in model_params.values()]).to(device=d)
    target_model_vector /= torch.norm(target_model_vector)

    # Prepare and normalize device models in a batch
    device_model_matrix = torch.stack([
        torch.cat([param.flatten() for param in device_list[nid].get_model().values()])
        for nid in neighbors
    ]).to(device=d)
    device_model_matrix /= device_model_matrix.norm(dim=1, keepdim=True)

    cosine_similarities = torch.matmul(device_model_matrix, target_model_vector)

    similarity_dict = {nid: similarity.item() for nid, similarity in zip(neighbors, cosine_similarities)}

    return similarity_dict

#NOTE NEW Co-Sim Functions
def compute_cosim_metric_from_state(state_dict1: dict, state_dict2: dict):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    # Iterate through the keys in the state dictionaries
    # for key in state_dict1:
    #     if key in state_dict2:    # Flatten the tensors and compute cosine similarity
    #         sim += abs(cos(state_dict1[key].flatten(), state_dict2[key].flatten()).item())
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


class Algorithm(Network):
    """ 
        Honestly... this could probably be added to Mobile Net Class.
        It is just another wrapper for a few abstract methdods hanlding community assignment and local aggregating.
    """
    
    def __init__(self,num_devices=25,num_classes=10,perceptual_map="turbo",threshold=10,star=False):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map,threshold=threshold)
        self.ap_param_stacks={}
        self.sf_seeds=max(2,int((.10)*num_devices))
        self.update_coordinates(0)
        self.star = star

    def run_global_round_setup_steps(self,round=0):
        print("Implement in child classes")
        print("Did Star flag work ", self.star)

        #NOTE: i think generally just contains these 2 steps
        
        # Simulate Movements
        self.update_coordinates(round) # If movement enabled
        
        # Re-linking Algorithm
        self.run_linking_algorithm() # Custom relinking methods
        
        # whatever else TODO before training
        if (self.star):
            N=len(self.device_list)
            self.ap_member_map={N:[i for i in range(N)]} #NOTE The trivial map which includes all
      
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
        """Just a default behavior.
            Pass "star" Flag to do nothing at this round (trivial behavior)
        """
        if (self.star):
            return
        self.build_proximity_graph(threshold=10)
        self.select_access_points_on_betweenness()
        # print(self.ap_member_map)

    def map_results_to_files(self,results):
        for (n,model,device) in results:
            # print(type(model), " " , len(model), " ", model[0], " ",model[2])
            torch.save(model, self.device_list[n].model_path)
    
    def assign_communities(self):
        """ Reset Communities After Organizing & After Training
                - ap_member_map: Key(AP) Value(List of Members)
                - ap_param_stacks: Directly Looks Up From Each Device Model
        """
        self.ap_member_map={ap:[] for ap in self.curr_apoints}

        for d in self.device_list:
            self.ap_member_map[d.parent_point].append(d.id)
        
        self.reset_colors()

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
class ProximityPreferentialAttachment(Algorithm):
    """ TODO:
        Set Hubs Based on Preferential Attachment mechanism but where it is also based on location at every round. 
        
        Essentially: Scale Free + Proximity

        NOTE: 
        This could be used as a first step and cosine reassignemnt could be used as a second. Or it could do well on its own. 
        Compare IID and Non-IID cases.
    """
    def __init__(self,num_devices=25,num_classes=10,perceptual_map="turbo",threshold=10):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map,threshold=threshold)
        ratio=.07
        self.sf_seeds=max(int(ratio*num_devices),2)
        return


    def run_global_round_setup_steps(self, round=0, threshold=10):

        self.update_coordinates(round)
        self.run_linking_algorithm(round_num=round, threshold=10)
        return

    def run_local_aggregation_round(self,max_iterations=3):
        return super().run_local_aggregation_round()


    def run_linking_algorithm(self,round_num=0,folder="static_updates/",save=False,weight="weight", threshold=10):
        """ Cases:
                - Static Devices/Static Topology
                - Static Device/Changing Topology
                - Mobile Devices/Changing Topology
        """
        self.run_proximity_preferential_attachment(threshold=threshold)
        # self.run_spatial_weighted_attachment()
        self.NXG1=copy.deepcopy(self.NXG2)
        self.select_access_points_on_betweenness()
        self.assign_communities()


    def run_proximity_preferential_attachment(self,threshold=.50,num_seeds=5,num_rounds=5,oporder=1):
        # G1 is a placeholder with which to build G2
        # G2 is initialized either with some seed nodes or minspantree
        self.NXG2=self.init_with_minspantree()
        #self.build_proximity_graph(threshold)
        self.fast_build_proximity_graph_(threshold=threshold)
        G=self.NXG1
        G2=self.NXG2 #NOTE/TODO -- maybe start with a min span Tree here or use the seeding method

        shuffler=copy.deepcopy(self.device_list)

        ## NOTE: Ordering 1:  Device THEN Degree
        if oporder == 1:
            random.shuffle(shuffler)
            for d in shuffler:
                # print(f"\nnext device {d.id}\n")
                neighbors=list(G.neighbors(d.id))
                if len(neighbors)==0:
                    continue
                degrees=[G2.degree(nid) for nid in neighbors]
                probas=degrees/np.sum(degrees)

                for attach in set(random.choices(neighbors, weights=probas, k=num_rounds)):
                    self.NXG2.add_edge(
                        d.id,attach, weight=self.Euclidean(self.device_list[d.id],self.device_list[attach] )
                )
            # print(probas)

        ## NOTE: Ordering 2 is Opposite --   Operations can Actually Lead to Different Distributions
        if oporder == 2:
            for _ in range(num_rounds):
                print(_,"\n")
                for d in self.device_list:
                    print(f"\nnext device {d.id}\n")
                    neighbors=list(G.neighbors(d.id))
                    if len(neighbors)==0:
                        print(f"skipped: {d.id}")
                        continue
                    degrees=[ max(G2.degree(nid)-1,1) for nid in neighbors]
                    # print(degrees)
                    probas=degrees/np.sum(degrees)
                    attach = random.choices(neighbors, weights=probas, k=1)[0]
                    # print(attach, " ", probas)
                    self.NXG2.add_edge(
                        d.id,attach, weight=self.Euclidean(self.device_list[d.id],self.device_list[attach] )
                    )

        # self.plot_topology(top=1)
        # plt.show()
        # self.plot_topology(top=2)
        # plt.show()

    def init_with_minspantree(self):
        A=np.ones([self.N,self.N])
        np.fill_diagonal(A, 0)
        temp=nx.from_numpy_array(A)
        temp=self.populate_edge_weights(temp)
        return nx.minimum_spanning_tree(
            temp,weight="weight",algorithm="kruskal")


    def seeding_algorithm(self,num_seeds=5):
        # TODO: run seeding out of the other file without need for copying code
        self.reset()
        nodes=set([i for i in range(self.N)])
        seeds,sampler = [],[]

        # seed
        for _ in range(num_seeds):
            seed=random.sample(list(nodes),k=1)[0]
            nodes.remove(seed)
            seeds.append(seed)

        # randomly connect
        for _ in range(np.random.randint(12,24)): # num pulls=np.random.randint(12,24)
            (i,j)=random.sample(seeds,k=2)
            self.add_edge(i,j),sampler.append(i),sampler.append(j)


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
        self.build_proximity_graph(threshold=threshold) # allow_isolates = False
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
            changes = 0
            for dobj in self.device_list:

                dvc = dobj.id
                if dvc in self.curr_apoints:  # APoints stay fixed
                    continue

                ## Similarity with "self" -- community model
                max_cosim = compute_model_to_community_cosim(self.device_list[dvc], self.device_list[dvc])
                max_nbr = dvc

                for nbr in self.NXG1.neighbors(dvc):
                    # Similarity with its neighbor's community
                    cosim = compute_model_to_community_cosim(self.device_list[dvc], self.device_list[nbr])
                    # Count Cost of Comparing
                    self.total_cost += self.Euclidean(self.device_list[dvc], self.device_list[nbr])
                    if cosim > max_cosim:
                        max_cosim, max_nbr = cosim, nbr
                self.device_list[dvc].parent_point = self.device_list[max_nbr].parent_point
                self.device_list[dvc].color = self.device_list[max_nbr].color
                
                # print(f"match? {dvc} : {max_nbr}, {max_cosim}")
                if (max_nbr != dvc):
                    changes += 1
            
        # Re-Set Communities after Comparisons
        self.assign_communities()
        return


class DPP(Algorithm):
    """
    Combines Proximity Preferential Attachment and Cosine Reassignment.
    """

    def __init__(self, num_devices=25, num_classes=10, perceptual_map="turbo", threshold=0.5):
        super().__init__(num_devices=num_devices, num_classes=num_classes, perceptual_map=perceptual_map,
                         threshold=threshold)
        self.sf_seeds = max(2, int(0.07 * num_devices))  # Seed value used in ProximityPreferentialAttachment
        self.ap_param_stacks = {}  # Parameter stack used in CosineReassignment

    def run_global_round_setup_steps(self, round=0, threshold=10):

        self.update_coordinates(round)
        self.run_linking_algorithm(round_num=round, threshold=self.threshold)

    def run_local_aggregation_round(self, max_iterations=3):

        self.compare_communities(max_iters=max_iterations)  # Cosine-based reassignment
        return super().run_local_aggregation_round()

    def run_linking_algorithm(self, round_num=0, threshold=0.5):

        self.run_spatial_weighted_attachment(threshold=threshold)  # Proximity part
        self.select_access_points_on_betweenness()
        #self.compare_communities()
        self.assign_communities()

    def run_spatial_weighted_attachment(self, threshold=0.75, num_rounds=6):
        self.NXG2 = self.init_with_minspantree()
        self.fast_build_proximity_graph_(threshold)
        G = self.NXG1
        G2 = self.NXG2

        for d in self.device_list:
            neighbors = list(G.neighbors(d.id))
            if not neighbors:
                continue
            cosine_sim = device_neighbor_similarity(d.get_model(), neighbors, self.device_list)
            # Calculate combined weights based on degree and spatial proximity
            combined_weights = [
                G2.degree(nid)*cosine_sim[nid]/ (self.Euclidean(self.device_list[d.id], self.device_list[nid]) + 1e-6)
                for nid in neighbors
            ]
            combined_weights = np.array(combined_weights) / np.sum(combined_weights)

            # Preferential attachment based on the combined metric
            for attach in random.choices(neighbors, weights=combined_weights, k=num_rounds):
                self.NXG2.add_edge(d.id, attach,
                                   weight=self.Euclidean(self.device_list[d.id], self.device_list[attach]))

    def init_with_minspantree(self):
        A=np.ones([self.N,self.N])
        np.fill_diagonal(A, 0)
        temp=nx.from_numpy_array(A)
        temp=self.populate_edge_weights(temp)
        return nx.minimum_spanning_tree(
            temp,weight="weight",algorithm="kruskal")

    def compare_communities(self, max_iters=10):

        # Precompute cosine similarities in batch
        # cosine_sim_matrix = batch_cosine_similarity(self.device_list)
        # cosine_sim_matrix.fill_diagonal_(-float('inf')) #Ignore diagonal


        # max_indices = torch.argmax(cosine_sim_matrix, dim=1)
        changes = 1000
        for iter in range(max_iters):
            if changes < 2:
                break
            else:
                changes = 0
            for dobj in self.device_list:

                dvc = dobj.id
                if dvc in self.curr_apoints:  # APoints stay fixed
                    continue

                ## Similarity with "self" -- community model
                max_cosim = compute_model_to_community_cosim(self.device_list[dvc], self.device_list[dvc])
                max_nbr = dvc

                for nbr in self.NXG2.neighbors(dvc):
                    # Similarity with its neighbor's community
                    cosim = compute_model_to_community_cosim(self.device_list[dvc], self.device_list[nbr])
                    # Count Cost of Comparing
                    self.total_cost += self.Euclidean(self.device_list[dvc], self.device_list[nbr])
                    if cosim > max_cosim:
                        max_cosim, max_nbr = cosim, nbr
                self.device_list[dvc].parent_point = self.device_list[max_nbr].parent_point
                self.device_list[dvc].color = self.device_list[max_nbr].color

                if (max_nbr != dvc):
                    changes += 1

        self.assign_communities()



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