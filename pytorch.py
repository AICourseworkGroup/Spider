import torch
from torch import nn
import os

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        """
        Initialize a neural network with configurable layers.
        
        Args:
            layer_sizes: A list of integers specifying the size of each layer.
                        Example: [784, 512, 256, 10] creates a network with:
                        - Input layer: 784 nodes
                        - Hidden layer 1: 512 nodes
                        - Hidden layer 2: 256 nodes
                        - Output layer: 10 nodes
        """
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Build the network layers dynamically based on layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # Add ReLU activation after each layer except the last one
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
