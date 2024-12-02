import copy
import multiprocessing

import networkx as nx
import torch
import os
from typing import List

from scipy.stats import uniform
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

    def __init__(self, num_classes: int, input_len: int) -> None:
        super(Net, self).__init__()
        self.channel_dim = 4
        if input_len == 3:
            self.channel_dim = 5
        self.conv1 = nn.Conv2d(input_len, 6, 5) # 32x32 -> 28x28
        self.pool = nn.MaxPool2d(2, 2) # 14x14
        self.conv2 = nn.Conv2d(6, 16, 5) #10x10 -> pool -> 5x5
        self.fc1 = nn.Linear(16 * self.channel_dim**2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.channel_dim**2)
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x))
        x = self.fc3(x1)
        return x, x1

def cal_uniform_act(out):
    zero_mat = torch.zeros(out.size()).to(out.device)
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)

    kldiv = nn.KLDivLoss(reduce=True)
    return kldiv(logsoftmax(out), softmax(zero_mat))


def local_train(cid, model, trainloader, valloder, parameters, cfg:DictConfig, device):

    if parameters is not None:
        model.load_state_dict(parameters)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])

    train(model, device, trainloader, optimizer, cfg["epochs"], cfg.beta, cfg.loss)
    metrics = {}
    
    return [cid, model.state_dict(), metrics]

def train(model, device, train_loader, optimizer, epochs, beta=10, loss_type='fedavg'):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    w_glob = copy.deepcopy(model.state_dict())
    l2_norm = nn.MSELoss()

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output, a1 = model(images)
            loss_ce = criterion(output, labels)

            if loss_type == 'fedmax':
                loss_kl = cal_uniform_act(a1)
                loss = loss_ce + beta * loss_kl
            elif loss_type == 'fedprox':
                reg_loss=0
                for name, param in model.named_parameters():
                    reg_loss += l2_norm(param, w_glob[name])
                loss = loss_ce + (beta/2) * reg_loss
            else:
                loss = loss_ce

            optimizer.zero_grad()
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
            outputs,_ = model(images)
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

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from tqdm import tqdm

def test_model(AP_state, testloaders, device, input_len):
    AP, state_dict = AP_state
    model = Net(10, input_len)  # Replace with your model creation function
    model.load_state_dict(state_dict)
    loss, accuracy = test(model, testloaders[AP], device)
    return AP, loss, accuracy

def update_ap_metrics(ap_avg_state_dict,testloaders,device, MAX_WORKERS=20, input_len=3):
    # Prepare the list of arguments for each process
    losses, accuracies = [], []
    ap_nodes=[]
    AP_states = list(ap_avg_state_dict.items())

    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {pool.submit(test_model, AP_state, testloaders, device, input_len): AP_state[0] for AP_state in AP_states}

    # Collect results as they complete
    for future in tqdm(as_completed(futures), total=len(futures), desc='Testing clients'):
        AP, loss, accuracy = future.result()
        ap_nodes.append(AP)
        losses.append(loss)
        accuracies.append(accuracy)
    pool.shutdown(wait=True)
    return {
        "ap_nodes": ap_nodes,
        "losses": losses,
        "accuracies": accuracies
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

def process_trainloaders(AP, trainloaders):
    class_counter = Counter()
    for trainloader in trainloaders:
        labels = []
        for _, batch_labels in trainloader:
            labels.extend(batch_labels.tolist())
        class_counter.update(labels)
    return len(trainloaders), AP, dict(class_counter)

def get_community_class_distributions(trainloaders, ap_member_map, MAX_WORKERS= 20):
    community_datasets = defaultdict(list)
    community_class_dist = {}
    for AP, members in ap_member_map.items():
        community_datasets[AP].extend([trainloaders[member] for member in members])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process_trainloaders, AP, loaders) for AP, loaders in community_datasets.items()]
        for future in tqdm(as_completed(futures), total=len(futures), desc='Community Distributions'):
            num_clients, AP, class_counts = future.result()
            community_class_dist[AP] = {
                "num_clients": num_clients,
                "Class_counts": class_counts
            }

    return community_class_dist


