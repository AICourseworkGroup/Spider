import random as rd
import math
import matplotlib.pyplot as plt

from genetic_algorithm import createTargetChromosome, createTargetChromosomeList, runGA
from neural_network_from_prac5 import Full_NN, genRanPoses
from genetic_algorithm import animateTargetChromosomes
from plot_spider_pose import plot_spider_pose
from pytorch import run_pytorch_nn
import matplotlib.pyplot as plt

def main():
    """
    Main function to generate target poses and train the neural network.
    Shows results/plots as it progresses through each step.
    """
    print("\n" + "="*60)
    print("STEP 1: Creating and Displaying Target Chromosomes")
    print("="*60)
    
    # Generate the target poses (GAPoses)
    targetChromosomeA = createTargetChromosome(math.radians(0), math.radians(-45), math.radians(-30), True)
    targetChromosomeB = createTargetChromosome(math.radians(20), math.radians(-45), math.radians(-30), False)
    
    # Generate full target walk cycle (this will display A and B)
    GATargetPoses = createTargetChromosomeList(targetChromosomeA, targetChromosomeB) # this also displays/plots the two poses A and B
    print(f"\nGenerated {len(GATargetPoses)} target frames for full walk cycle.")

    print("\n" + "="*60)
    print("STEP 2: Animate Target Chromosomes")
    print("="*60)

    choice = input("Do you want to animate the target chromosomes? (y/n): ").lower()
    if choice == 'y':
        print("Animating target chromosomes...")
        animateTargetChromosomes("Target Chromosomes", GATargetPoses, delay=0.01) 
        print("Animation complete.")

    print("\n" + "="*60)
    print("STEP 3: Running Genetic Algorithm")
    print("="*60)

    maxGenerations = int(input("Enter the maximum number of generations for the GA (Recommended: 300): "))
    populationSize = int(input("Enter the population size for the GA (Recommended: 300): "))
    mutationRate = float(input("Enter the mutation rate for the GA (Recommended 0.01): "))

    print("\nRunning GA to match target poses...")
    GAPoses = runGA(maxGenerations, populationSize, mutationRate, GATargetPoses)
    print("GA Complete!")

    print("\n" + "="*100)
    print("STEP 4: Animate GA Generated Poses")
    print("="*100)
    print("Animating GA-generated poses...")
    animateTargetChromosomes("GA Chromosome", GAPoses, delay=0.1)
    print("Animation complete.")

    # Generate random poses for neural network training
    num_poses = len(GAPoses)
    print(f"\nGenerating {num_poses} random input poses for neural network training...")
    inputData = genRanPoses(popSize=num_poses)
    print("Input data generated.")

    print("\n" + "="*100)
    print("STEP 5: Training Custom Neural Network")
    print("="*100)
    nn = Full_NN(X=24, HL=[512, 256], Y=24)
    print("Neural Network initialized: 24 -> 512 -> 256 -> 24")
    print("\nTraining custom NN (printing progress)...")
    nn.train_nn(x=inputData, target=GAPoses, epochs=1000, lr=0.01)
    print("Custom NN training complete!")

    print("\n" + "="*100)
    print("STEP 6: Display Custom NN Input")
    print("="*100)
    testPose = genRanPoses(popSize=1)[0]
    print("Generated random test pose. Displaying...")
    plot_spider_pose(testPose, title="Custom NN - Test Input")
    plt.pause(2)
    plt.close()

    print("\n" + "="*100)
    print("STEP 7: Display Custom NN Output")
    print("="*100)
    predictedPose = nn.FF(testPose)
    print("Custom NN prediction generated. Displaying...")
    plot_spider_pose(predictedPose, title="Custom NN - Prediction Output")
    plt.pause(2)
    plt.close()

    print("\n" + "="*100)
    print("STEP 8: Training PyTorch Neural Network")
    print("="*100)
    pytorch_model = run_pytorch_nn(inputData, GAPoses, epochs=300, lr=0.01)
    print("PyTorch NN training complete!")
    
    print("\n" + "="*100)
    print("STEP 9: Display PyTorch NN Input")
    print("="*100)
    testPose2 = genRanPoses(popSize=1)[0]
    print("Generated random test pose. Displaying...")
    plot_spider_pose(testPose2, title="PyTorch NN - Test Input")
    plt.pause(3)
    plt.close()
    
    print("\n" + "="*100)
    print("STEP 10: Display PyTorch NN Output")
    print("="*100)
    pytorch_prediction = pytorch_model.predict(testPose2)
    print("PyTorch NN prediction generated. Displaying...")
    plot_spider_pose(pytorch_prediction, title="PyTorch NN - Prediction Output")
    plt.pause(3)
    plt.close()

    print("\n" + "="*100)
    print("ALL STEPS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()