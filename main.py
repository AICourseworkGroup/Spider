import math

from genetic_algorithm import createTargetChromosome, createTargetChromosomeList
from helpers import display_all_steps_complete, display_creating_target_chromosomes, display_custom_nn_input, display_custom_nn_output, display_generating_input_data, display_animating_ga_generated_poses, display_pytorch_nn_input, display_pytorch_nn_output, display_training_pytorch_nn, train_custom_nn, generate_random_pose, run_genetic_algorithm


def main():
    """
    Main function to generate target poses and train the neural network.
    Shows results/plots as it progresses through each step.
    """
    display_creating_target_chromosomes()

    #=========== Genetic Algorithm Steps =============
    
    # Target poses (GAPoses)
    targetChromosomeA = createTargetChromosome(math.radians(0), math.radians(-45), math.radians(-30), True)
    targetChromosomeB = createTargetChromosome(math.radians(20), math.radians(-45), math.radians(-30), False)
    
    # Walk cycle (display A and B)
    GATargetPoses = createTargetChromosomeList(targetChromosomeA, targetChromosomeB)
    
    # Get GA parameters from user
    maxGenerations, populationSize, mutationRate = display_generating_input_data(GATargetPoses)

    # Run Genetic Algorithm
    GAPoses = run_genetic_algorithm(maxGenerations, populationSize, mutationRate, GATargetPoses)

    # Display GA generated poses
    display_animating_ga_generated_poses(GAPoses)
    
    # Generate random poses
    inputData = generate_random_pose(GAPoses)
    
# ============ Neural Network Training =============

    # Train custom NN
    nn = train_custom_nn(inputData, GAPoses)

    # Display custom NN input
    testPose = display_custom_nn_input()

    # display_custom_nn_output(nn, testPose)
    display_custom_nn_output(nn, testPose)

    # Train PyTorch NN
    pytorch_model = display_training_pytorch_nn(inputData, GAPoses)
    
    # Display PyTorch NN input
    testPose2 = display_pytorch_nn_input()
    
    # Display PyTorch NN output
    display_pytorch_nn_output(pytorch_model, testPose2)

    # Display all steps complete message
    display_all_steps_complete()

if __name__ == "__main__":
    main()