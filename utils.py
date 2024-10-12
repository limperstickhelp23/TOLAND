import os

import torch


def train(model, trainloader, optimizer, epochs, device):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    print(f"Before :{model.state_dict()}")
    model.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    print(f"After :{model.state_dict()}")


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

directory = "aggregation_metrics"

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)
def save_metrics(round_number, accuracy, loss):
    # Define the file path for this round
    file_path = os.path.join(directory, f"round_{round_number}.txt")

    # Save the metrics to the file
    with open(file_path, "w") as file:
        file.write(f"Round: {round_number}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Loss: {loss}\n")
