"""
Functions for training and testing models.

Jacob J  Oct 4 2024

"""

import torch.nn  as nn
import torch.nn.functional as F
import torch




def train(model, train_loader, optimizer, epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for _ in range (epochs):
        for image, label in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += F.cross_entropy(output, label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
        accuracy =  correct / len(test_loader.dataset)
        return accuracy, test_loss

