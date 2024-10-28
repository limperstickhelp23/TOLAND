from netsim.network import *
import os,random
import numpy as np
import networkx as nx
import torch.nn as nn
import torch

#UTILS
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

#ALGORITHMS
class StaticServer(MobileNet): 
    def __init__(self,num_devices=20):
        super().__init__(n=num_devices)
        self.sf_seeds=int((.10)*num_devices)
        self.outdir="static_updates/"
        self.mobile=False

    def rewire_round(self,round_num=0,folder="static_updates/",save=False,weight="weight"):
        """ The Static Algorithm will simply generate a new selected
            topology type at each round, but without any device movement.
        """
        self.set_sf_topology(num_seeds=self.sf_seeds)
        self.generate_assignments(weight=weight)
        self.attribute_communities()
        if (save):
            self.assignments_to_yaml(os.path.join(folder,f"round_{round_num}.yaml"))
        
class EfficientLessCentralizedServer(MobileNet):
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

class MinSpanTreeServer(MobileNet):
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
        self.generate_assignments(weight=self.weight)
        self.attribute_communities()
        # if (save):
        #     self.assignments_to_yaml(os.path.join(folder,f"round_{round_num}.yaml"))

class MaxEntropyServer(MobileNet):
    def __init__(self,num_devices=20):
        super().__init__(n=num_devices)   

## NOTE: Add New Algorithms here
class MyNewAlgorithm(MobileNet):
    def __init__(self,num_devices=20):
        super().__init__(n=num_devices)   
    
    def rewire_round(self):
        return

class CosineReassignment(MobileNet):
    """
        Goal: Try to reassign devices to other communities based on cosine similarity
        but try to do so while remaining cost-efficient. 
        
        Naive Idea:
            - First Round everyone trains and the community models are aggregated
            - Then everyone gets back the "community model"
            - Assuming they have a copy of their own local model, next they compare their
            own local model to each of their neighbors community models (and their own)
            - Then they reassign themselves to the community with most similarity
            - Then they broadcast their desired community to devices around them
            - Any common communities will form a link from proximity neighbors and if none then they will remain isolated
            - The last problem to solve now is flow of information 

        Within Local
    """

    def __init__(self,num_devices,num_classes,perceptual_map):
        super().__init__(num_devices=num_devices,num_classes=num_classes,perceptual_map=perceptual_map)
        self.ap_param_stacks={} # TODO: don't need this?
    
    def rewire_round(self,round_num=0):
        self.build_proximity_graph()
        self.set_custom_topology(self.A)
        self.plot_topology()
    
    def init_community_models(self):
        for (ap,dvcs) in self.ap_member_map.items():
            self.device_list[ap].community_model.load_state_dict(self.device_list[ap].model.state_dict())
            for dvc in dvcs:
                self.device_list[dvc].community_model.load_state_dict(self.device_list[ap].model.state_dict())
    
    def assign_communities(self):
        """
        Simulated Version: (Same as Integrated Version ?)
            Re-Set Communities After Organizing & After Training 
                ap_member_map: Key(AP) Value(List of Members)
                ap_param_stacks: Directly Looks Up From Each Device Model
        """
        self.ap_member_map={ap:[] for ap in self.curr_apoints}
        self.ap_param_stacks={ap:[] for ap in self.curr_apoints}
        
        for d in self.device_list:
            self.ap_member_map[d.parent_point].append(d.id)
            self.ap_param_stacks[d.parent_point].append(d.model.state_dict()) #NOTE: can use .parameters() or .state_dict()
        
        self.reset_colors() # NOTE: whoops forgot this before

    def local_aggregation_round(self):
        """
        Simulated Version:
            The local aggregation round maps parameters internally.
            This way each device knows its community parameters.
        """
        ap_avg_state_dict={} #NOTE: temp variable to match back semantically to cloud.py
        for (AP,members) in self.ap_member_map.items():
            community_state_dict=aggregate_params(self.ap_param_stacks[AP])
            self.device_list[AP].model.load_state_dict(community_state_dict)
            for idx in members:
                self.device_list[idx].community_model.load_state_dict(community_state_dict)
            ap_avg_state_dict[AP]=community_state_dict
        # self.ap_params=ap_avg_state_dict
        
        return ap_avg_state_dict
    
    def map_results(self,results):
        """
        For Integration:
            Since we train externally, this function facilitates mappings (either stored in the object or in files)
            between the Algorithm internal representation of devices to the externally trained models
        
        """
        results={r[0]:r[1] for r in results} # convert to dictionary
        for (AP, peers) in self.ap_member_map.items():
            for peer in peers:
                self.device_list[peer].model.load_state_dict(results[peer]) # attribute results to devices
        return

    
    def integrated_local_aggregation_round(self):

        """
        Integrated Version:
            Because the communities have been re-assigned. We now need to 
            redo the internal mappings.
            
            Aggregate AP makes the initial community models after local training.

            Wait. But probably makes the most sense to keep the community models fixed until the end of
            all the local training rounds

            Instead facilitate mappings (with function above). Intermediate rounds can just be swaps of information
            and local training. 
            
            TODO: figure out facilitating local aggregation rounds. The reason for this is it might be better
            to run re-assignment prior to doing any local aggregation so that local aggregation rounds can be even better/
            more accurate.

            Then, the last local aggregation round can be used to communicate back up to server for central round
        """

        for (AP,members) in self.ap_member_map.items():
            community_state_dict=aggregate_params(self.ap_param_stacks[AP])
            self.device_list[AP].model.load_state_dict(community_state_dict)
            for idx in members:
                self.device_list[idx].community_model.load_state_dict(community_state_dict)
            # self.ap_params[AP]=aggregate_params(chosen_params)
        return
    
    #TODO: figure out how to connect results back to local aggregation algorithm
    def ap_aggregate(self):
        """
        Integrated Version:
            * Map results list to the Algorithm local copy of parameters
            *TODO: streamline this -- need a better solution

            *TODO: Instead of this accepting the results list, it will just do a local
            aggregation round based ont eh current mappings
        """
        results=dict(results)

        for (AP, peers) in self.ap_member_map.items():
            for (AP,members) in self.ap_member_map.items():
                community_state_dict=aggregate_params(self.ap_param_stacks[AP])
                self.ap_params[AP]=community_state_dict
        

    def compare_communities(self,max_iters=10):
        """
            NOTE: In this case, devices store parameters from both their own model and the community model.
            Then they compare their model to the community models of their neighbors to decide whether the neighbor's 
            community is a better fit. The baseline is cosine similarity with their current community model.
            If no community is better now switch occurs
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
                max_cosim=compute_device_to_community_cosim(self.device_list[dvc],self.device_list[dvc])
                max_nbr,max_color=dvc,self.device_list[dvc].color
            
                for nbr in self.NXG1.neighbors(dvc):
                    
                    # Similarity with it's neighbor's community
                    cosim=compute_device_to_community_cosim(self.device_list[dvc], self.device_list[nbr])
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
    
    def self_assign(self,max_iters=10):
        """
        NOTE: FIRST VERSION  -- New version usese comparisons to community models instad.
       
        This works ok except sometimes the order can affect it. 
        So for example (0,3,6,9) are all supposed to be red but 9 is initially yellow.
        Then 3 was similar to 9 so was pulled into yellow camp and brought 6 with it. 

        This is at the first round. To get around this, I hope doing community aggregations will help.

        Another way would be to try to assign them to the most similar access point. However they may not always be close
        to the access point with the best similarity
        """

        changes=1000

        for _ in range(max_iters):
            if changes < 2:
                break
            else:
                changes = 0
            

            for dobj in self.device_list:
                
                dvc=dobj.id
                if dvc in self.curr_apoints:
                    continue # NOTE: access points stay fixed, I think this will be key to algorithm but could be wrong
                
                ## Similarity with "self" -- community model
                max_cosim=compute_device_to_community_cosim(self.device_list[dvc],self.device_list[dvc])
                max_nbr,max_color=dvc,self.device_list[dvc].color

                for nbr in self.NXG1.neighbors(dvc):

                    cosim=compute_device_model_cosim_metric(self.device_list[dvc], self.device_list[nbr])
                    if cosim > max_cosim:
                        max_cosim,max_nbr,max_color=cosim,nbr,self.device_list[nbr].color
                        self.device_list[dvc].parent_point = self.device_list[nbr].parent_point

                # print(f"match? {dvc} : {max_nbr}, {max_cosim}")
                if max_color != self.device_list[dvc].color:
                    changes += 1
                
                self.device_list[dvc].color = max_color   
        # TODO/NOTE: load current access points from devices -- shoul dbe its own function  
        # Re-Set Communities after Organization
        self.assign_communities()
        
        return

