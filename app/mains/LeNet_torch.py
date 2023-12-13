import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys

# Define a function to get intermediate outputs for the convolutional layers

def get_intermediate_output(model, layer_name, data):
    model.eval()
    outputs = []
    for layer in model.features:
        data = layer(data)
        if layer_name in str(layer):
            outputs.append(data)
    return outputs

class LeNet(nn.Module):
    def __init__(self, input_channels):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main(train_num, test_num, datasets):
    print('Number of training samples:', train_num)
    block_size = 5
    stride = 1

    if datasets == 'MNIST' or datasets == 'FMNIST':
        if datasets == 'MNIST':
            transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        if datasets == 'FMNIST':
            transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
            test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        channel = 1
    elif datasets == 'CIFAR10':
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        channel = 3

    train_loader = DataLoader(train_dataset, batch_size=train_num, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_num, shuffle=True)

    model = LeNet(input_channels = channel)

    #model summary
    print(model)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

    # Get intermediate outputs for convolutional layers
    block_outputs = get_intermediate_output(model, 'Conv2d', images)
    for i, output in enumerate(block_outputs):
        print(f"Block {i+1} output shape:", output.shape)

if __name__ == '__main__':
    num_train = int(sys.argv[1])
    num_test = int(sys.argv[2])
    datasets = sys.argv[3]

    main(num_train, num_test, datasets)
