"""
Jacob J  Oct 4 2024

"""
from asyncio import gather
from collections import OrderedDict
from typing import Dict, List

import asyncio
import flwr
import numpy as np
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate

from models import train
from models import test


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
        self.peers = peers
        self.model = instantiate(model_cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.train_loader = trainloader
        self.valloader = valloader
        self.device_id = int(device_id)
        self.ready_event = asyncio.Event()
        self.AggregationRegistry = {peer.device_id: asyncio.Event for peer in peers}

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config: Dict[str, Scalar]):
        asyncio.run(self.fit_concurrent(parameters, config))

    async def fit_concurrent(self, parameters, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        alpha = config['alpha']
        lr = config['lr']
        momentum = config['momentum']
        epochs = config['epochs']
        self.peers = config['peers']
        for _ in range(alpha):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
            train(self.model, train_loader=self.train_loader, optimizer=optimizer, epochs=epochs, device=self.device)
            self.ready_event.set()
            await self.wait_for_peers()
            await self.device_aggregation()
            await self.wait_for_peer_aggregation()
            for event in self.AggregationRegistry.values():
                event.clear()
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
        neighbor_params = {idx : peer.get_parameters({ }) for idx, peer in enumerate(self.peers)}

        avg_param = []
        for param in zip(*neighbor_params.values()):
            #Using FedAvg
            avg_param.append(np.mean(param, axis=0))

        await gather(*[peer.set_device_status(peer.device_id) for peer in self.peers])
        return avg_param

    def set_device_status(self, device_id):
        self.AggregationRegistry[device_id].set()

    async def wait_for_peer_aggregation(self)->None:
        await gather(*[event.wait() for event in self.AggregationRegistry.values()])

    async def wait_for_peers(self)->None:
        await gather(*[peer.ready_event.wait() for peer in self.peers])


def generate_client_fn(trainloaders, valloaders, model_cfg):
    def client_fn(cid):
        return Honey2Device(cid, trainloaders[cid], valloaders[cid], model_cfg, [ ]).to_client()

    return client_fn










