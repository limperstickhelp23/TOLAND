from collections import defaultdict

import numpy as np
import os

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

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

