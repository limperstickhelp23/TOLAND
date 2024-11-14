import networkx as nx
import torch
import os
from typing import List

from tqdm import tqdm

from deprecate.client import test
import pandas as pd
import numpy as np
from collections import Counter

import torch
from torch import nn
import torch.nn.functional  as F

from collections import defaultdict

import numpy as np
import os

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

import torch
from omegaconf import DictConfig

from netsim.network import Device


###NOTE: CLIENT MODELS
class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int, input: int) -> None:
        super(Net, self).__init__()
        self.channel_dim = 4
        if input == 3:
            self.channel_dim = 5
        self.conv1 = nn.Conv2d(input, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * self.channel_dim**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.channel_dim**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def local_train(cid, model, trainloader, valloder, parameters, cfg:DictConfig, device):

    if parameters is not None:
        model.load_state_dict(parameters)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    train(model, device, trainloader, optimizer, cfg["epochs"])
    metrics = {}
    
    return [cid, model.state_dict(), metrics]

def train(model, device, train_loader, optimizer, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

def test(model, testloader, device):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset) if (len(testloader.dataset) > 0) else 0.0
    return loss, accuracy




##################################
# AGGREGATION METHODS
##################################
def aggregate_params(model_params):
    
    averaged_state_dict = {}
    for key in model_params[0].keys(): #conv1_
        param_stack = torch.stack([state_dict[key] for state_dict in model_params], dim=0)
        avg_params = torch.mean(param_stack, dim=0)
        averaged_state_dict[key] = avg_params

    return averaged_state_dict

def aggregate_at_server(ap_avg_state_dict):
    return aggregate_params(ap_avg_state_dict)

def ap_aggregate(results, ap_routes)-> List[torch.Tensor]:
    ap_params = {}
    for AP, peers in ap_routes.items():
        chosen_params = [result[1] for result in results if result[0] in peers]

        avg_ap_state_dict = aggregate_params(chosen_params)
        ap_params[AP] = avg_ap_state_dict

    return ap_params

def get_parameters(ap_state_dict, ap_routes, node_id):
    if ap_state_dict is None:
        return None
    key = None
    for AP, peers in ap_routes.items():
        if  node_id in peers:
            key = AP
            break

    return ap_state_dict[key]

def get_ap_metrics(ap_avg_state_dict, path, server_round, a, model, testloaders, device):
    file_path = path+f'access_point_eval_{server_round}/aggregate_{a}.txt'

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for AP, state_dict in ap_avg_state_dict.items():
            model.load_state_dict(state_dict)
            loss, accuracy = test(model, testloaders[AP], device)
            file.write(f"{AP} : \n\tloss: {loss} \n\taccuracy: {accuracy}\n")

def update_ap_metrics(ap_avg_state_dict,model,testloaders,device):
    losses,accuracies=[],[]
    for (AP, state_dict) in tqdm(ap_avg_state_dict.items(), desc='Testing clients'):
        model.load_state_dict(state_dict),
        loss, accuracy = test(model, testloaders[AP], device)
        losses.append(loss)
        accuracies.append(accuracy)
    
    return {
        "ap_nodes":list(ap_avg_state_dict.keys()),
        "losses":losses, 
        "accuracies":accuracies
    }

##################################
# MISCELLANEOUS METHODS
##################################
def normalize_dict(d):
    D=np.sum([v for v in list(d.values())])
    for (k,v) in d.items():
        d[k]=np.round(v/D,3)
    return d

def get_class_distribution(dataloader):
    class_counts = Counter()
    for _, labels in dataloader:
        class_counts.update(labels.tolist())
    return dict(class_counts)

def check_iidness(dataloader):
    d=normalize_dict(get_class_distribution(dataloader))
    for (k,v) in d.items():
        print(f"{k}: {v}")

def Euclidean(d1: Device, d2: Device):
    """NOTE: probably just rewrite this at the Network level to update distances every global round"""
    return np.sqrt(
        np.linalg.norm(np.array([d1.x, d1.y]) - np.array([d2.x, d2.y]))
    )



