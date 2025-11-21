import random as rd
import math
import matplotlib.pyplot as plt

from genetic_algorithm import createTargetChromosome, createTargetChromosomeList, runGA
from neural_network import Full_NN, genRanPoses
from plot_spider_pose import plot_spider_pose
from pytorch import run_pytorch_comparison
import matplotlib.pyplot as plt

def main():
    """
    Main function to generate target poses and train the neural network.
    """
    # Generate the target poses (GAPoses)
    # These will be what the genetic algorithm aims to match for each chromosome.
    # These are also the ideal poses we want the neural network to learn.
    targetChromosomeA = createTargetChromosome(math.radians(0), math.radians(-45), math.radians(-30), True)
    targetChromosomeB = createTargetChromosome(math.radians(20), math.radians(-45), math.radians(-30), False)
    print("Running Genetic Algorithm to generate target poses...")
    GAPoses = createTargetChromosomeList(targetChromosomeA, targetChromosomeB)
    print("Target poses generated.")

    # Next we run the genetic algorithm. We ask the user to enter their own
    # parameters for the GA. We supply our own recommended values that we believe
    # give the best results. We need lots of generations and a high population
    # to ensure the best fitness possible for each generated chromosome
    maxGenerations = int(input("Enter the maximum number of generations for the GA (Recommended: 300): "))
    populationSize = int(input("Enter the population size for the GA (Recommended: 300): "))
    mutationRate = float(input("Enter the mutation rate for the GA (Recommened 0.01): "))

    runGA(maxGenerations, populationSize, mutationRate, GAPoses)

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
    nn = Full_NN(X=24, HL=[24, 24], Y=24)
    print("Neural Network initialized.")

    # Train the Neural Network
    # The inputData is a set of random poses, and GAPoses is the target.
    # The network will learn to transform the random poses into the target poses.
    print("Training Neural Network...")
    nn.train_nn(x=inputData, target=GAPoses, epochs=1000, lr=0.05)
    print("Neural Network training finished.")

    #test the neural network with a new random pose
    print("Testing Neural Network with a new random pose...")
    testPose = genRanPoses(popSize=1)[0]
    predictedPose = nn.FF(testPose)
    print("Test Pose (Input):")
    print(testPose)
    print("Predicted Pose (Output):")
    print(predictedPose)
    plot_spider_pose(testPose, title="Test Pose (Input)")
    plt.pause(3)  # Display for 3 seconds
    plt.close()   # Close the figure
    
    plot_spider_pose(predictedPose, title="Predicted Pose (Output)")
    plt.pause(3)  # Display for 3 seconds
    plt.close()   # Close the figure
    
    # Run PyTorch comparison with the same data
    print("\n--- Running PyTorch Comparison ---")
    run_pytorch_comparison(inputData, GAPoses, epochs=1000, lr=0.001)
    

if __name__ == "__main__":
    main()