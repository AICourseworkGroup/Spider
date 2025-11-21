import torch
from torch import nn
import matplotlib.pyplot as plt

from neural_network import genRanPoses
from plot_spider_pose import plot_spider_pose

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device.upper()} device")


class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes=[24, 512, 256, 24]):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_nn(model, input_data, target_data, epochs=1000, lr=0.001):
    """Train the neural network with input and target data."""
    # Convert to tensors
    X = torch.tensor(input_data, dtype=torch.float32)
    y = torch.tensor(target_data, dtype=torch.float32)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    model.train()
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
    
    return model


def test_nn(model, input_data, target_data):
    """Test the trained neural network and visualize results."""
    model.eval()
    with torch.no_grad():
        # Test on a random pose
        testPose = genRanPoses(popSize=1)[0]
        testTensor = torch.tensor(testPose, dtype=torch.float32).unsqueeze(0)
        prediction = model(testTensor).squeeze(0).numpy().tolist()
    
    # Show results
    print("\nTest Input:", testPose[:6], "...")
    print("Prediction:", prediction[:6], "...")
    
    plot_spider_pose(testPose, title="Input Pose")
    plt.pause(3)
    plt.close()
    
    plot_spider_pose(prediction, title="Predicted Pose")
    plt.pause(3)
    plt.close()


# This function is meant to be called from main.py with existing data
def run_pytorch_comparison(input_data, target_data, epochs=1000, lr=0.001):
    """
    Simple function to train and test PyTorch NN with existing data.
    
    Args:
        input_data: List of input poses from main.py
        target_data: List of target poses (GAPoses) from main.py
        epochs: Number of training epochs
        lr: Learning rate
    """
    print("\n" + "="*50)
    print("PyTorch Neural Network Comparison")
    print("="*50)
    
    model = NeuralNetwork(layer_sizes=[24, 512, 256, 24])
    print(f"Training on {len(input_data)} poses...")
    
    model = train_nn(model, input_data, target_data, epochs, lr)
    print("\nTesting...")
    test_nn(model, input_data, target_data)
    
    print("="*50 + "\n")