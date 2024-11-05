import torch
import os
from typing import List
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


###NOTE: CLIENT MODELS
class Net(nn.Module):
    """A simple CNN suitable for simple vision tasks."""

    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
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




###NOTE: FL Aggregator Functions 
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
    for (AP, state_dict) in ap_avg_state_dict.items():
        model.load_state_dict(state_dict),
        loss, accuracy = test(model, testloaders[AP], device)
        losses.append(loss)
        accuracies.append(accuracy)
    
    return {
        "ap_nodes":list(ap_avg_state_dict.keys()),
        "losses":losses, 
        "accuracies":accuracies
    }

### NOTE: DATA LOADING 
def get_mnist(data_path: str = "/Users/jacobjoseph/GitHub/TOLAND/data"):
    """Download MNIST and apply minimal transformation."""
    data_path=os.path.dirname(os.path.abspath(__file__))
    print(f"MNIST Downloaded to {data_path}")

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    bool_ = False
    if not os.path.exists(os.path.join(data_path, "MNIST")):
        bool_ = True

    trainset = MNIST(data_path, train=True, download=bool_, transform=tr)
    testset = MNIST(data_path, train=False, download=bool_, transform=tr)

    return trainset, testset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, iid: bool = True, alpha=0.5):
    """Prepare data loaders for each client and choose to non-iid or iid datasets"""

    # download MNIST in case it's not already in the system
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    # Calculate base size and remainder
    num_images = len(trainset) // num_partitions
    remainder = len(trainset) % num_partitions

    # Initialize partition_len with base size for each partition
    partition_len = [num_images] * num_partitions

    # Distribute remainder across the first few partitions
    for i in range(remainder):
        partition_len[i] += 1

    if iid:
        print("??")
        trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))
    else:
        trainsets = dirichlet_partition(trainset, partition_len, alpha)

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )
        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader


def dirichlet_partition(trainset, partition_len, alpha=0.5, num_classes=10):
    """
    dirichlet partition
    :param num_classes:
    :param trainset: train data set
    :param partition_len: # of datasets to be created
    :param alpha: lower the value the more the non-iid of the resulting datasets
    :return: a list of partitioned datasets of varying distributions
    """
    class_indices = defaultdict(list)

    for idx, (image, label) in enumerate(trainset):
        class_indices[label].append(idx)

    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Use Dirichlet distribution to partition indices non-IID
    partitions = [[] for _ in range(len(partition_len))]
    for label in range(num_classes):
        class_size = len(class_indices[label])

        proportions = np.random.dirichlet([alpha] * len(partition_len))
        proportions = np.array([int(p * class_size) for p in proportions])

        proportions[-1] = class_size - proportions[:-1].sum()

        current_idx = 0
        for i, partition in enumerate(partitions):
            partition.extend(class_indices[label][current_idx:current_idx + proportions[i]])
            current_idx += proportions[i]

    # Create Subsets for each partition
    partitioned_datasets = [Subset(trainset, indices) for indices in partitions]

    return partitioned_datasets


### NOTE: MISCELLANEOUS
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



