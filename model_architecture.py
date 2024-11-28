import torch
import torch.nn as nn
import torch.nn.functional as F

class FruitClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitClassifierCNN, self).__init__()

        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Conv layer 1 (RGB image)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Conv layer 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)       # Max pooling
        self.fc1 = nn.Linear(64 * 25 * 25, 512)  # Fully connected layer 1 (flattened image size)
        self.fc2 = nn.Linear(512, num_classes)   # Output layer
        self.relu = nn.ReLU()                    # ReLU activation
        self.dropout = nn.Dropout(0.5)           # Dropout for regularization

    def forward(self, x):
        # Apply convolution, ReLU activation, and max pooling
        x = self.pool(self.relu(self.conv1(x)))  # First conv layer
        x = self.pool(self.relu(self.conv2(x)))  # Second conv layer
        
        # Flatten the output to feed into fully connected layers
        x = x.view(-1, 64 * 25 * 25)  # Reshape the tensor to match input size of fc1
        
        # Apply fully connected layers and dropout
        x = self.relu(self.fc1(x))    # First fully connected layer
        x = self.dropout(x)           # Dropout
        x = self.fc2(x)               # Output layer (class probabilities)

        return x
