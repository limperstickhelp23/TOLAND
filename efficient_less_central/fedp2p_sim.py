from configs import *
from node import Node, load_datasets
import networkx as nx
from random import sample
from typing import List
import numpy as np
from models.simpleCNN import Net


def init(num=10):    #NOTE: INIT to see that Node() is working, set all parameters equal to device id
    devices=[]
    for i in range(num):
        n=Node(i)
        net=Net() #Random
        x=n.get_parameters(net)
        dummies=[]
        for layer in x:
            dummies.append(np.ones_like(layer)*i)
        n.dummy_init(net,value=i)
        devices.append(n)

    def check_init():
        for d in devices:
            print("Check Weights are Set: ", d.weights[1], "\n" ,d.weights[-1], "\n")

    return devices

NUM_CLIENTS=10
ROUNDS=10
COMMUNITY_SIZE = 3
DEVICES=init(NUM_CLIENTS)
#TODO: custom paritionining, here I make paritions outside the loop
FDS = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})



class Cloud:
    def __init__(self, client_ids=[]):
        self.client_ids=client_ids
        self.model_template=np.ones(1) # TODO
        self.communities = []

        self.server_model=None # local copy of its weights (last round essentially)
    
    def sample_clients(self,L):
        #NOTE: this just randomly samples participants at each round
        return sample(self.client_ids,L)
    

    def make_communities(self,N):
        #TODO: more intelligent community formation
        #N = size of community
        clients = [i for i in self.client_ids]
        comms=[]
        while len(clients) >= N:
            s=sample(clients,N)
            comms.append(s)
            clients=list(set(clients)-set(s))
        comms.append(clients) # stragglers
        self.communities=comms

        return comms

    def send_model(self):
        #NOTE/TODO : something like this basically
        for c in self.communities:
            AP=sample(c,1)[0]
            c.remove(AP)


    def FedAvg(self,client_params: List,client_weights=[]):
        
        # make a copy
        old=self.server_model
        new=[]
        new_layer=np.array([])

        for (client,params) in enumerate(client_params):
            for (loc,layer) in params:
                assert (layer == np.Ndarray)
                if client == 0:
                    new_layer=old[loc]
                else:
                    new_layer=np.vstack(new_layer, layer)
                new[loc]=new_layer
        # finally average everything
        if client_weights == []:
            for (loc,layer) in enumerate(new):
                new[loc]=np.mean(layer,axis=0)

        #NOTE: or something like that  (basically FedAvg)

        return new

# Weight 
def pool_weights(friends,biases=[]):
    
    # make a copy
    new=[]
    for i in range(len(friends[0].weights)):
        new.append( np.vstack([ biases[j]*friend.weights[i] for (j,friend) in enumerate(friends)]))
    
    return [np.sum(new[i],axis=0)/np.sum(biases) for i in range(len(new))] #pooling


def train_communities():
    #TODO -- here is where we want parallelism
    return None

def cluster_agg(community,fds: FederatedDataset):

    #NOTE Basically this is a prototype of FedP2P
    sizes=[]
    print("aggregating new community: ", community)
    for member in community:
        net=Net() # load architecture
        trainloader, valloader, testloader =  load_datasets(fds=fds,partition_id=member)
        sizes.append(len(trainloader.dataset))
        DEVICES[member].set_parameters(net) #NOTE: stored at the node here
        print(f"Member {member} simulate some training... beep. boop. converged")
        DEVICES[member].freeze_parameters(net)
    
    subset=[DEVICES[member] for member in community]
    new=pool_weights(subset, biases=sizes)

    # print("Check pooling operation: ", new[1])

    return new

#NOTE: here is a basic sim
        
cloud=Cloud([d.pid for d in DEVICES])

for round in range(ROUNDS):
    # ids=cloud.sample_clients(L=4)
    # print("Sample: " , ids)
    comms=cloud.make_communities(COMMUNITY_SIZE)
    models=[]
    print(f"\nNEW ROUND {round}, new communities: ", comms,"\n")
    
    # Local Round #TODO: another spot for parallelism
    for (n,comm) in enumerate(comms):
        models.append(cluster_agg(comm,fds=FDS))

    print("see fedp2p result:")
    for m in models:
        print(m[1])
    
    
    # Global Round
    #TODO


    





