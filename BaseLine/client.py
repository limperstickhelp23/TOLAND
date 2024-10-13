import torch
from functorch.dim import Tensor
from omegaconf import DictConfig


def local_train(cid, model, trainloader, valloder, parameters, cfg:DictConfig, device):
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    momentum = cfg["momentum"]

    if parameters is not None:
        model.load_state_dict(parameters)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train(model, device, trainloader, optimizer, epochs)

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
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
