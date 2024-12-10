from collections import defaultdict

import numpy as np
import os

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from omegaconf import DictConfig


def get_mnist(data_path: str = "./data"):

    """Download MNIST and apply minimal transformation."""

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    bool_ = not os.path.exists(os.path.join(data_path, 'MNIST'))

    trainset = MNIST(data_path, train=True, download=bool_, transform=tr)
    testset = MNIST(data_path, train=False, download=bool_, transform=tr)
    return trainset, testset

def get_cifar10(data_path: str = "./data"):

    data_path = os.path.join(data_path, 'CIFAR10')
    tr = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    bool_ = not os.path.exists(data_path)
    if bool_:
        os.makedirs(data_path, exist_ok=True)

    trainset = CIFAR10(root=data_path, train=True, download=bool_, transform=tr)
    testset = CIFAR10(root=data_path, train=False, download=bool_, transform=tr)

    return trainset, testset


def prepare_dataset(cfg : DictConfig, val_ratio: float = 0.05):
    """Prepare data loaders for each client and choose to non-iid or iid datasets"""
    Dataset = cfg.dataset
    num_partitions = cfg.num_partitions
    batch_size = cfg.batch_size
    iid = cfg.iid
    alpha = cfg.alpha
    # download MNIST in case it's not already in the system
    if Dataset == "MNIST":
        trainset, testset = get_mnist()
    else:
        trainset, testset = get_cifar10()

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

    try:
        if iid:
            trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2025))
        else:
            trainsets = dirichlet_partition(trainset, partition_len, alpha)
    except ValueError as ve:
        print(f"Details: {ve}")
        print(f"Error occurred in random_split: length of trainset: {len(trainset)} \n partition_len: {partition_len}.")
    except Exception as e:
        print("An unexpected error occurred.")
        print(f"Error type: {type(e).__name__}")
        print(f"Details: {e}")
        trainsets = None  # Optional fallback logic

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2025)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
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

if __name__ == "__main__":
    get_cifar10()