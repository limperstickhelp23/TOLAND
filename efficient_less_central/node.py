from models import simpleCNN
from models.simpleCNN import Net
import numpy as np
from numpy.random import uniform
import torch
from flower.imports import *
import copy


class Node(NumPyClient):
    """
        A client node. 
    """
    def __init__(self,PID):
        self.x = uniform(XBDS)
        self.y = uniform(YBDS)
        self.coords=(self.x,self.y)
        self.pid = PID
        # self.model = simpleCNN.Net() #NOTE: probably everyone usese same model
        self.weights: List[np.ndarray] = []
        # print("Fill In Movement Model")
    
    def dummy_init(self,net, value=1):
        self.weights = [value*np.ones_like(arr) for arr in net.state_dict().values()]
    
    #NOTE: this comes stright from tutorial -- conveneince for getting params 
    def set_parameters(self,net):
        params_dict = zip(net.state_dict().keys(), copy.deepcopy(self.weights))
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    
    def freeze_parameters(self,net):
        """ NOTE: alternatively we could save in files to relieve memory?
        """
        self.weights = [val.cpu().numpy() for _, val in net.state_dict().items()]

    def get_parameters(self,net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def todo(self):
        print("TODO: Fill In Movement Model")
        print("Fill in Communication Model")
        print("Set up stats collecter : counter, cost, etc.")

def node_test():
    n=Node(1)
    print(n.pid)
    print(n.weights)
    net=simpleCNN.Net() 
    t=n.get_parameters(net)
    for item in t:
        print("Layer: ", item.shape)
    # print([type(item) for item in t]) # NOTE: see torch docs on state dict, I think it 
    # print([item.shape for item in t]) # NOTE: I think it is layer by layer

# class FlowerClient(NumPyClient):
#     def __init__(self, partition_id, net, trainloader, valloader):
#         self.partition_id = partition_id
#         self.net = net #NOTE: 
#         self.trainloader = trainloader
#         self.valloader = valloader

#     def get_parameters(self, config):
#         print(f"[Client {self.partition_id}] get_parameters")
#         return get_parameters(self.net)

#     def fit(self, parameters, config):
#         print(f"[Client {self.partition_id}] fit, config: {config}")
#         set_parameters(self.net, parameters)
#         train(self.net, self.trainloader, epochs=1)
#         return get_parameters(self.net), len(self.trainloader), {}

#     def evaluate(self, parameters, config):
#         print(f"[Client {self.partition_id}] evaluate, config: {config}")
#         set_parameters(self.net, parameters)
#         loss, accuracy = test(self.net, self.valloader)
#         return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def load_datasets(fds: FederatedDataset, partition_id: int):
    # TODO: look into data management -- I think FDS handles paritions for us
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


def demo_trainloader():
    trainloader, _, _ = load_datasets(partition_id=0)
    batch = next(iter(trainloader))
    images, labels = batch["img"], batch["label"]

    # Reshape and convert images to a NumPy array
    # matplotlib requires images with the shape (height, width, 3)
    images = images.permute(0, 2, 3, 1).numpy()

    # Denormalize
    images = images / 2 + 0.5

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show()

# def client_fn(context: Context) -> Client:
#     net = Net().to(DEVICE)
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
#     return FlowerClient(partition_id, net, trainloader, valloader).to_client()
