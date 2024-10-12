"""
Jacob J  Oct 4 2024

"""
from asyncio import gather
from collections import OrderedDict
from typing import Dict, List, Tuple

import asyncio
import flwr
import numpy as np
import torch
from colorama import Back
from flwr.client import Client
from flwr.common import NDArrays, Scalar, FitRes, Context
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import Status
from hydra.utils import instantiate
from torch import Code

from utils import train
from utils import test


#Framework for Device to device training/aggregation
class Honey2Device(flwr.client.NumPyClient):
    """Decentralized communications
    @device_id - id of device set by cloud
    @tau - # training rounds
    @alpha - # Aggregation rounds
    @peers - set of neighbors of type flwr.ClientProxy
    """
    def __init__(self, device_id, trainloader, valloader, model_cfg, peers: List['Honey2Device']):
        super().__init__()
        self.selected_peers = None
        self.AP = None
        self.peers = peers
        self.model = instantiate(model_cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.train_loader = trainloader
        self.valloader = valloader
        self.device_id = int(device_id)
        self.ready_event = asyncio.Event()

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['epochs']
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, trainloader=self.train_loader, optimizer=optimizer, epochs=epochs, device=self.device)
        with open('output.txt', 'w') as file:
            # Write content to the file
            file.write(f'params of {self.device_id} : {self.model.state_dict()}\n')
        return self.get_parameters(self.model), len(self.train_loader), {}


    async def fit_concurrent(self, parameters, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        alpha = config['alpha']
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['epochs']
        self.selected_peers = list(config['peers'])
        self.selected_peers.remove(self.device_id)
        self.AP = config['AP']
        with open('output.txt', 'w') as file:
            # Write content to the file
            file.write(f'training: {self.device_id}')
        for _ in range(alpha):
            self.ready_event.clear()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            train(self.model, trainloader=self.train_loader, optimizer=optimizer, epochs=epochs, device=self.device)

            if self.AP != self.device_id: # you are not an Access point
                self.ready_event.set()
                await self.peers[self.AP].ready_event.wait()
            else:
                await self.wait_for_peers()
                await self.device_aggregation() #AP aggregate
                self.ready_event.set()
            self.ready_event.clear()

        return self.get_parameters(self.model), len(self.train_loader), { }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        # do local evaluation (call same function as centralised setting)
        accuracy, loss = test(self.model, self.valloader, device=self.device)
        # send statistics back to the server

        return float(loss), len(self.valloader), {"accuracy": accuracy}



    async def device_aggregation(self):
        neighbor_params = [peer.get_parameters({}) for peer in self.peers if peer.device_id in self.selected_peers]
        with open('output.txt', 'w') as file:
            # Write content to the file
            file.write(f'neighbor_params of {self.device_id} : {neighbor_params}\n')
        avg_param = []
        for param in zip(*neighbor_params):
            #Using FedAvg
            avg_param.append(np.mean(param, axis=0))

        return avg_param

    #wait until all chosen peers are done training
    async def wait_for_peers(self)->None:
        await gather( *[peer.ready_event.wait() for peer in self.peers if peer.device_id in self.selected_peers])



def generate_client_fn(trainloaders, valloaders, model_cfg):
    def client_fn(context: Context)->Client:
        cid = context.node_config['partition-id']
        return Honey2Device(int(cid), trainloaders[int(cid)], valloaders[int(cid)], model_cfg, [ ]).to_client()

    return client_fn

    def client_fn_2(context: Context) -> Client:
        net = Net()
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
        return FlowerClient(partition_id, net, trainloader, valloader).to_client()










