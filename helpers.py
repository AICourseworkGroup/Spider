
from spider_pose import plot_spider_pose
from genetic_algorithm import animateTargetChromosomes, runGA
from nn_self import genRanPoses, Full_NN
from nn_pytorch import run_pytorch_nn
from matplotlib import pyplot as plt


#Main helper functions for displaying steps in main.py
def display_creating_target_chromosomes():
    print("\n" + "="*60)
    print("STEP 1: Creating and Displaying Target Chromosomes")
    print("="*60)

def display_generating_input_data(GATargetPoses):
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

    return maxGenerations, populationSize, mutationRate

def run_genetic_algorithm(maxGenerations, populationSize, mutationRate, GATargetPoses):
    print("\nRunning GA to match target poses...")
    GAPoses = runGA(maxGenerations, populationSize, mutationRate, GATargetPoses)
    print("GA Complete!")
    return GAPoses

def display_animating_ga_generated_poses(GAPoses):
    print("\n" + "="*100)
    print("STEP 4: Animate GA Generated Poses")
    print("="*100)
    print("Animating GA-generated poses...")
    animateTargetChromosomes("GA Chromosome", GAPoses, delay=0.1)
    print("Animation complete.")
    return animateTargetChromosomes

def generate_random_pose(GAPoses):
    num_poses = len(GAPoses)
    print(f"\nGenerating {num_poses} random input poses for neural network training...")
    inputData = genRanPoses(popSize=num_poses)
    print("Input data generated.")
    return inputData

# ============ Custom Neural Network Training =============

def train_custom_nn(inputData, GAPoses):
    print("\n" + "="*100)
    print("STEP 5: Training Custom Neural Network")
    print("="*100)
    
    nn = Full_NN(X=24, HL=[512, 256], Y=24)
    print("Neural Network initialized: 24 -> 512 -> 256 -> 24")
    print("\nTraining custom NN (printing progress)...")
    nn.train_nn(x=inputData, target=GAPoses, epochs=1000, lr=0.01)
    print("Custom NN training complete!")
    return nn

def display_custom_nn_input():
    print("\n" + "="*100)
    print("STEP 6: Display Custom NN Input")
    print("="*100)
    testPose = genRanPoses(popSize=1)[0]
    print("Generated random test pose. Displaying...")
    testPose = genRanPoses(popSize=1)[0]
    plot_spider_pose(testPose, title="Custom NN - Test Input")
    plt.pause(2)
    plt.close()
    return testPose

def display_custom_nn_output(nn, testPose):
    print("\n" + "="*100)
    print("STEP 7: Display Custom NN Output")
    print("="*100)
    predictedPose = nn.FF(testPose)
    print("Custom NN prediction generated. Displaying...")
    plot_spider_pose(predictedPose, title="Custom NN - Prediction Output")
    plt.pause(2)
    plt.close()
    return predictedPose

def display_training_pytorch_nn(inputData, GAPoses):
    print("\n" + "="*100)
    print("STEP 8: Training PyTorch Neural Network")
    print("="*100)
    pytorch_model = run_pytorch_nn(inputData, GAPoses, epochs=300, lr=0.01)
    print("PyTorch NN training complete!")
    return pytorch_model

def display_pytorch_nn_input():
    print("\n" + "="*100)
    print("STEP 9: Display PyTorch NN Input")
    print("="*100)
    testPose2 = genRanPoses(popSize=1)[0]
    print("Generated random test pose. Displaying...")
    plot_spider_pose(testPose2, title="PyTorch NN - Test Input")
    plt.pause(3)
    plt.close()
    return testPose2

def display_pytorch_nn_output(pytorch_model, testPose2):
    print("\n" + "="*100)
    print("STEP 10: Display PyTorch NN Output")
    print("="*100)
    pytorch_prediction = pytorch_model.predict(testPose2)
    print("PyTorch NN prediction generated. Displaying...")
    plot_spider_pose(pytorch_prediction, title="PyTorch NN - Prediction Output")
    plt.pause(3)
    plt.close()
    return pytorch_prediction

def display_all_steps_complete():
    print("\n" + "="*100)
    print("ALL STEPS COMPLETE")
    print("="*100)