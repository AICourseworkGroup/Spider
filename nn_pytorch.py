import torch
import torch.nn as nn
import numpy as np

class PyTorch_NN(nn.Module):
    
    def __init__(self, input_size=24, hidden_sizes=[512, 256], output_size=24):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features (24 angles)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output features (24 angles)
        """
        super(PyTorch_NN, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        # Stack the hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # turn list of input layers, hidden layers, ReLU Layers and output layer into a neural network
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)
    
    def train_network(self, input_data, target_data, epochs=100, lr=0.001):
        """
        Train the neural network.
        
        Args:
            input_data: List of input poses (random poses)
            target_data: List of target poses (GA poses)
            epochs: Number of training epochs
            lr: Learning rate
        """
        # Convert data to tensors
        x_train = torch.tensor(input_data, dtype=torch.float32)
        y_train = torch.tensor(target_data, dtype=torch.float32)
        
        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            predictions = self(x_train)
            loss = criterion(predictions, y_train)
            
            # Backward pass for loss calculations
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
        
        print(f"Training complete. Final loss: {loss.item():.6f}")
        return loss.item()
    
    def predict(self, input_pose):
        """
        Make a prediction from a single pose.
        
        Args:
            input_pose: Single input pose (list of 24 angles)
        Returns:
            Predicted pose (list of 24 angles)
        """
        # Set the model to evaluation mode (disables dropout/batchnorm if present)
        self.eval()
        # Disable gradient calculation (faster, saves memory, not needed for inference)
        with torch.no_grad():
            # Convert input pose (list or array) to a PyTorch tensor of type float32
            x = torch.tensor(input_pose, dtype=torch.float32)
            # Pass the input through the network to get the prediction
            prediction = self(x)
            # Convert the output tensor to a Python list
            return prediction.numpy().tolist()


def run_pytorch_nn(input_data, target_data, epochs=50, lr=0.01):
    """
    Create, train, and test a PyTorch neural network.
    
    Args:
        input_data: List of random input poses
        target_data: List of target GA poses
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model
    """
    print("\n=== PyTorch Neural Network ===")
    
    # Create model
    model = PyTorch_NN(input_size=24, hidden_sizes=[512, 256], output_size=24)
    print(f"Model created with architecture: 24 -> 512 -> 256 -> 24")
    
    # Train model
    print("\nTraining PyTorch network...")
    model.train_network(input_data, target_data, epochs=epochs, lr=lr)
    
    return model
