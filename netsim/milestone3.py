from .basealgorithm import *
from networkx.algorithms.community import kernighan_lin_bisection
from sklearn.cluster import KMeans
from copy import deepcopy


class ModularDPP(Algorithm):
    """
    IDEA: After Selecting AP's at the first hierarchy level (keep it at 5), we then look for even better sub-communities using modularity at the next level. 

    START HERE
    The edge weights are DPP (inspired) calculations with parameters λ1 and λ2. 
    λ1 -- controls the degree at which to favor distance in weight
    λ2 -- controls the degree at whcih to favor weight similarity
    
    LATER
    Late but possible additions: 
        * Infrequent similarity updates ("Color Memory Mechanism")
            * Each device keeps a binary vector of last round's communtiy involvement
            * Dot product will therefore either be 0 or 1 between neighbors
        
        * Alternatively:
            * Weighted voting system either by (accuracies is possible)
            * Or using communtiy vector which is occasionally updated (every 3 rounds or so)
    """

    def __init__(self, num_devices=25, num_classes=10, perceptual_map="turbo", threshold=0.5):
        super().__init__(num_devices=num_devices, num_classes=num_classes, perceptual_map=perceptual_map,threshold=threshold)
        self.NXG2 = nx.from_numpy_array(self.A)
        self.sf_seeds = max(2, int(0.07 * num_devices))  # Seed value used in ProximityPreferentialAttachment
        self.ap_param_stacks = {}  # Parameter stack used in CosineReassignment
        self.cosim_matrix = [[None for _ in range(100)] for _ in range(100)]
        self.num_apoints = 5
        self.clusters = None
        self.reset_cost_collection()
        self.num_communication_rounds=2
        
        import warnings
        warnings.filterwarnings("ignore")    # NOTE: ignoring this torch warning keeps cluttering output
    
    def reset_cost_collection(self):
        self.level_1_total_edge_cost=0
        self.level_2_total_edge_cost=0
        self.server_out_cost=0 # 5 access points
        self.server_in_cost=0  # 10 access points
    
    def calculate_server_round_cost(self,star=False): 
        ### NOTE: Overriding parent Cost Calculations Because It's a bit different here with multiple layers here, star is dummy var here
        self.server_round_costs.append(
            self.level_1_total_edge_cost + self.server_out_cost + self.server_in_cost + (self.num_communication_rounds)*self.level_2_total_edge_cost
        )
        round_cost=self.server_round_costs[-1]
        self.TOTAL_COST += round_cost
        return round_cost

    def most_central_node_rule(self,sg):
        """ Node with minimum sum of euclidean distance to its neighbors (in other words it minimizes a cluster cost)
        """
        sg_costs=np.sum(nx.to_numpy_array(sg),axis=1)
        nodes=list(sg)
        ap=nodes[np.argmin(sg_costs)]
        self.curr_apoints.append(ap)
        self.ap_member_map[ap]=nodes
  
        k = 0
        for n in nodes:
            if n != ap:
                k += self.D[ap,n]
        
        return np.min(sg_costs)

    def run_global_round_setup_steps(self, round=0, threshold=10, plot=False):
        
        self.reset_cost_collection()
        self.update_coordinates(round)
        self.generate_distances() # NOTE: important to have access to distances at every round
        self.run_linking_algorithm(round_num=round, threshold=self.threshold,plot=plot)
        # self.run_community_division_step(round_num=round,plot=plot)

    def run_local_aggregation_round(self, num_communication_rounds=3):
        return super().run_local_aggregation_round()

    
    def run_linking_algorithm(self, round_num=0, threshold=0.5,plot=False):
        """ NOTE: See picture. I think running betweeness is inherently inefficient due to the clustered
            layout of real world mobility. We want better spreading out of nodes. Either:

             - Idea 1: Run PPA + group nodes on hubs + closest paths
             - Idea 2: Run modularity directly on the distance based graph  ( either K-means or Spectral seem to be good choices, or modularity + threshold)

            #NOTE Init with MinSpanTree ( Not the Best Idea Perhaps ) --> See Screenshot as to Why
                # G=nx.from_numpy_array(self.D)
                # self.NXG1=nx.minimum_spanning_tree(G, weight="weight")
                # self.select_access_points_on_betweenness(switch=False)
        """

        def plot_topology(G : nx.graph, pause=1.5, edge_weights=.2):
            nx.draw(G, with_labels=True,font_color="white",pos=self.get_positions(),width=edge_weights,node_size=1)
            for (ii,(k,v)) in enumerate(self.ap_member_map.items()):
                cluster=list(v)
                cluster.remove(k)
                nx.draw_networkx_nodes(G,nodelist=cluster, node_color= COLORS[ii], pos=self.get_positions())   
            nx.draw_networkx_nodes(G,nodelist=self.curr_apoints,node_color="black",node_shape='*',node_size=800,pos=self.get_positions())
            plt.draw(),plt.pause(pause),plt.cla()


        ## NOTE : Instead Try Using K-Means for Distance
        kmeans = KMeans(n_clusters=self.num_apoints)
        labels = kmeans.fit_predict(np.array(self.get_positions())) # data=np.array(self.get_positions())
        
        # NOTE: make adajacency from kmeans clusters
        C=np.zeros([len(labels),self.N])
        for lbl in labels:
            C[lbl,:]=(np.array(labels) == lbl)*1
        self.NXG1=nx.from_numpy_array( (C.T@C - np.eye(self.N))*self.D )
    
        #NOTE: MST way # self.NXG1=nx.minimum_spanning_tree(self.NXG1)    # clusters=list(nx.connected_components(self.NXG1))
        self.ap_member_map,self.curr_apoints= {}, []
        subgraphs=[self.NXG1.subgraph(cluster).copy() for cluster in list(nx.connected_components(self.NXG1))]
        
        for sg in subgraphs:
            #NOTE: Use Distance Based Hub Selection
            mincost=self.most_central_node_rule(sg)
            self.level_1_total_edge_cost += mincost #NOTE: cost required for AP to talk to its members (at first level) 
            #NOTE:  MST way could work fine too  # hub,score=betweeness_rule(sg,top=1)  # self.curr_apoints.append(hub[0])
        
        self.server_out_costs=np.sum([self.server_distance(self.device_list[ii]) for ii in self.ap_member_map.keys()])
       
        # for (k,v) in self.ap_member_map.items():
        #     print(k, " ", v)

        if (plot):
            plot_topology(self.NXG1)
    
        
        ## NOTE: Second Step (Nested Definition) Divides Communities Into 2 Additional Sub-Communities
        def run_community_division_step(round_num=0,rate=1,cutoff=.8,seed_rounds=3,plot=True,plot_func=plot_topology):   # True  False

            def sigmoid(x,k):
                return 1 / (1 + np.exp(-k* (x-seed_rounds)))

            # NOTE Cosimilarity Schedule -- enforcing Convex combination is maybe too weak of a signal (??)
            schedule=[0.1,0.1,0.1,0.2,0.333,0.5,0.6,0.67,0.7,0.8] # NOTE: manual schedule
            if round_num < seed_rounds:
                λ1=0.0
            else:
                λ1=min(sigmoid(round_num,rate),cutoff)
            λ2=(1-λ1)
            # print("Check Weight : " , λ1, λ2)

            prior_map=deepcopy(self.ap_member_map)
            self.ap_member_map={}

            for (k,v) in prior_map.items():
                nodes=list(v)
                
                #NOTE: edge-scoring algorithm based on DPP
                subset=[self.device_list[node] for node in nodes]
                S=batch_cosine_similarity(subset).cpu().numpy()      #NOTE: deal w/ inf values:  P=np.nan_to_num(P, nan=0, posinf=pmax, neginf=0) # Deal with Inf values
                P=1/(self.D[np.ix_(nodes,nodes)]+1e-10) # add tiny epsilon to ensure bounded values
                np.fill_diagonal(P,0.0),np.fill_diagonal(S,0.0) # NOTE: erase biasfrom self-node scores
                S/=np.sum(S)
                P/=np.sum(P) #NOTE: Simple Step is to normalize matrices -- more complicated would be to convert to doubly-stochastic matrix)
    
                A = λ1*S + λ2*P            
                np.fill_diagonal(A,0.0)

                """ NOTE:
                    -> Here we could either run some type of preferential attachment (but I think it's redundant).
                    -> Or instead, use the fully connected topology the AP runs a "DPP-inspired" modularity (Kern-Lin) division.
                """
                G=nx.Graph()
                G = nx.from_numpy_array(A)
                mapping = {i: nodes[i] for i in range(len(nodes))}
                G = nx.relabel_nodes(G, mapping)
                
                try:
                    #NOTE: For some edge case I haven't figured out yet -- this might fail -- I think the edge case is Singleton Node from K Means
                    partition = kernighan_lin_bisection(G,weight="weight")  #NOTE : might be preferable because it ensures 2 groups exactly
                except:
                    partition=list(G)
                
                # partition = greedy_modularity_communities(G, weight='weight',resolution=1.0)  # NOTE: resolution is super sensitive (24 communtiies)
                subgraphs=[G.subgraph(cluster).copy() for cluster in partition]

                for (n,sg) in enumerate(subgraphs):
                    _=self.most_central_node_rule(sg)  #NOTE: Use Distance Based Hub Selection for AP's
                
                self.server_in_cost=np.sum([self.server_distance(self.device_list[ii]) for ii in self.ap_member_map.keys()])

            self.curr_apoints = list(self.ap_member_map.keys()) #NOTE: Reset New Access Points (there are 5-10 now)

            ## NOTE: Build NXG2 which represents the final topology after all divisive steps
            
            self.NXG2 = nx.Graph()
            self.NXG2.add_nodes_from([ii for ii in range(self.N)])
            for (k,v) in self.ap_member_map.items(): 
                for n in v:
                    if n != k:
                        self.NXG2.add_edge(k,n, weight=self.D[k,n])

            if (plot):
                plot_func(self.NXG1)
                plot_func(self.NXG2, pause=1 , edge_weights=.2)
            
            # Extract all edge weights as a list
            weights = [data['weight'] for _, _, data in self.NXG2.edges(data=True)]

            # Sum the weights
            self.level_2_total_edge_cost = sum(weights)

            run_community_division_step(round_num=round_num,plot=plot)
            

            #____ Possible TODO______________
            #(1) NOTE -- for visualizing community structure
            #(2) NOTE -- compute the cost of the modularity algorithm (at least just for communications)
            #(3) NOTE: we will treat it as if it is an additional agg round basically
            #(4) NOTE -- use a different cosim function (like DPP algorithm above)
        























##NOTE: MARK FOR SCRATCH  -- These are All Old Ideas
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
###NOTE: Other Ideas | Can Ignore | Prototype New Ideas Etc. 
# class EfficientLessCentralizedServer(Algorithm):
#     def __init__(self,num_devices=20):
#         super().__init__(n=num_devices) 
#         self.mobile=False   
#         self.ap_color="black"
    
#     def rewire_round(self,round_num=0):
#         indices=[i for i in range(self.N)]
#         clusters=random.sample([i for i in range(self.N)], self.num_apoints)
#         indices=list(set(indices)-set(clusters))
#         self.curr_apoints=list(clusters)
#         memberships={}

#         for dvc in indices:
#             head=random.sample(clusters,1)[0]
#             memberships[dvc]=head
#             self.routes[dvc]=[dvc,head]
#             self.A[dvc,head]=self.A[head,dvc]=1
        
#         self.set_custom_topology(self.A)
#         self.reset_colors()
#         self.attribute_communities(memberships)

# class MinSpanTreeServer(Algorithm):
#     def __init__(self,num_devices=20,λ=10):
#         super().__init__(n=num_devices,λ=λ)
#         self.mobile=True
#         self.weight="weight"
#         self.save=False
#         self.outdir="mst_tree_updates/"
#         self.method="fully_connected"
    
#     def rewire_round(self,round_num=0):

#         if self.method == "fully_connected":
#             A=np.ones([self.N,self.N])
#             np.fill_diagonal(A, 0)
#             self.set_custom_topology(A)
#         elif self.method == "proximity":
#             self.build_proximity_graph()
#             self.set_custom_topology(self.A)
#         else:
#             print("method not recognized")
#             return
    
#         self.set_custom_topology(
#             nx.adjacency_matrix(
#                 nx.minimum_spanning_tree(self.NXG1,weight=self.weight,algorithm="kruskal")
#             )
#         )
#         # self.generate_assignments(weight=self.weight)
#         self.attribute_communities()
#         # if (save):
#         #     self.assignments_to_yaml(os.path.join(folder,f"round_{round_num}.yaml"))

# class MaxEntropyServer(Algorithm):
#     def __init__(self,num_devices=20):
#         super().__init__(n=num_devices)   

# ## NOTE: Add New Algorithms here
# class MyNewAlgorithm(Algorithm):
#     def __init__(self,num_devices=20):
#         super().__init__(n=num_devices)   
    
#     def rewire_round(self):
#         return




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