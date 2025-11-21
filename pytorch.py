import torch
import torch.nn as nn
import numpy as np

class PyTorch_NN(nn.Module):
    """Simple PyTorch neural network for spider pose generation"""
    
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
        
        # Add hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Add output layer (no activation for regression)
        layers.append(nn.Linear(prev_size, output_size))
        
        # Combine all layers
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
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
        
        print(f"Training complete. Final loss: {loss.item():.6f}")
        return loss.item()
    
    def predict(self, input_pose):
        """
        Make a prediction on a single pose.
        
        Args:
            input_pose: Single input pose (list of 24 angles)
            
        Returns:
            Predicted pose (list of 24 angles)
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(input_pose, dtype=torch.float32)
            prediction = self(x)
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
