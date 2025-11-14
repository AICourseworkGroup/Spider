from genetic_algorithm import run_ga
from neural_network import Full_NN, genRanPoses

def main():
    """
    Main function to run the genetic algorithm and train the neural network.
    """
    # Run the genetic algorithm to get the target poses (GAPoses)
    # These are the ideal poses we want the neural network to learn.
    print("Running Genetic Algorithm to generate target poses...")
    GAPoses = run_ga(generations=100, populationSize=100, mutationRate=0.1)
    print("Genetic Algorithm finished.")

    # Generate random poses to be used as input for the neural network.
    # The number of input poses should match the number of target poses.
    num_poses = len(GAPoses)
    print(f"Generating {num_poses} random poses for NN input...")
    inputData = genRanPoses(popSize=num_poses)
    print("Input data for NN generated.")

    # Initialize the Neural Network
    # Input layer (X) and output layer (Y) size should be 24 (for 24 angles in a pose).
    # Hidden layers (HL) can be configured as needed.
    print("Initializing Neural Network...")
    nn = Full_NN(X=24, HL=[12, 12], Y=24)
    print("Neural Network initialized.")

    # Train the Neural Network
    # The inputData is a set of random poses, and GAPoses is the target.
    # The network will learn to transform the random poses into the target poses.
    print("Training Neural Network...")
    nn.train_nn(x=inputData, target=GAPoses, epochs=100, lr=0.05)
    print("Neural Network training finished.")

if __name__ == "__main__":
    main()
