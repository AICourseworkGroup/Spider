from genetic_algorithm import generate_target_poses
from neural_network import Full_NN, genRanPoses
from plot_spider_pose import plot_spider_pose

def main():
    """
    Main function to generate target poses and train the neural network.
    """
    # Generate the target poses (GAPoses)
    # These are the ideal poses we want the neural network to learn.
    print("Generating target poses...")
    GAPoses = generate_target_poses()
    print("Target poses generated.")

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
    plot_spider_pose(predictedPose, title="Predicted Pose (Output)")
    

if __name__ == "__main__":
    main()