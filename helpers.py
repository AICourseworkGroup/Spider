from genetic_algorithm import animateTargetChromosomes

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