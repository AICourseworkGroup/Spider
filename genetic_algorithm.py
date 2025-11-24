import random as rd
import math
import matplotlib.pyplot as plt
from spider_pose import plot_spider_pose

def createTargetChromosome(a, b, c, isA):
    """
        Handles chromosome encoding we first looked at the problem of the spider by assigning all 8 
        legs to individual vectors consisting of 3 individual joints a, b, c that are grouped into 
        legs where left legs L1, L2, L3, L4, while the right legs are Grouped as R4, R3, R2, R1. 
        The function params are from the main function where we define the angles for joints a, b, c.
        
        Args:
            a: Angle for joint a in radians
            b: Angle for joint b in radians
            c: Angle for joint c in radians
            isA: Boolean indicating if the chromosome is of type A or B
        """
    if isA:
        l1, l2, l3, l4 = [a, b, c], [a, b, c], [a, b, c], [a, b, c]
        r4, r3, r2, r1 = [-a, b, c], [-a, b, c], [-a, b, c], [-a, b, c]
        
    else:
        l1, l2, l3, l4 = [a, b, c], [-a, b, c], [a, b, c], [-a, b, c]
        r4, r3, r2, r1 = [-a, b, c], [a, b, c], [-a, b, c], [a, b, c]


    # return a flat Python list (24 floats) in radians
    angles = l1 + l2 + l3 + l4 + r4 + r3 + r2 + r1

    print(f"Target chromosome: {angles} ") 

    return angles

def createTargetChromosomeList(targetChromosomeA, targetChromosomeB):
    """
        This function creates a list of target chromosomes that represent a walk cycle between two 
        target chromosomes A and B. 
        
        Args:
            targetChromosomeA: The first target chromosome (standing pose)
            targetChromosomeB: The second target chromosome (stepping pose)"""

    print("Target A")
    title = "Target Chromosome A"
    plot_spider_pose(targetChromosomeA, title=title)
    plt.pause(2)  # Display for 2 seconds
    plt.close()   # Close the figure
    

    title = "Target Chromosome B"
    plot_spider_pose(targetChromosomeB, title=title)
    plt.pause(2)  # Display for 3 seconds
    plt.close()   # Close the figure

    # Step 1: Take away the difference between the two chromosomes.
    differenceBetweenChromosomes = [b - a for a, b in zip(targetChromosomeA, targetChromosomeB)]

    # Step 2: Divide this difference by 149 to get the amount we need to increment by.
    differenceBetweenChromosomes = [d / 149 for d in differenceBetweenChromosomes]

    # Here we initialise the list of target chromosomes with the initial standing one.
    targetChromosomes = [targetChromosomeA]

    # Step 3: For loop that creates the 149 inbetween chromosomes. It will do this by adding 
    # the difference / 149 to the angles everytime and then appending that new target chromosome to the list.
    for i in range(149):
        step = i + 1
        currentTargetChromosome = [a + step * d for a, d in zip(targetChromosomeA, differenceBetweenChromosomes)]

        # After calculating the incremented frame we append it to the list before looping again
        targetChromosomes.append(currentTargetChromosome)

    # Step 4: We will now take away instead of adding to complete the 2nd half of the walk cycle
    for i in range(149):
        step = i + 1
        currentTargetChromosome = [a - step * d for a, d in zip(targetChromosomeB, differenceBetweenChromosomes)]

        # After calculating the incremented frame we append it to the list before looping again
        targetChromosomes.append(currentTargetChromosome)

    # With all intermediary frames added, we can now add the final (300th) frame
    targetChromosomes.append(targetChromosomeA)

    # This is a list of every target chromosome for all 300 frames. Every time we generate a new target we 
    # append it to this list. When we run the genetic algorithm we'll do a for loop where we go to the next 
    # target chromosome every time that i increments.
    
    return targetChromosomes


# This is a function we use to generate a random angle in radians. We call this to avoid redundant code.
def randRadianGen():
    angle = math.radians(rd.randint(-180, 180))
    return angle

# In this function we create a random population for the genetic algorithm. We call randRadianGen to generate 
# each angle. 

def createRandomPopulation(populationSize):
    population = []

    for i in range(populationSize):
        l1 = [randRadianGen(), randRadianGen(), randRadianGen()]
        l2 = [randRadianGen(), randRadianGen(), randRadianGen()]
        l3 = [randRadianGen(), randRadianGen(), randRadianGen()]
        l4 = [randRadianGen(), randRadianGen(), randRadianGen()]
        r4 = [randRadianGen(), randRadianGen(), randRadianGen()]
        r3 = [randRadianGen(), randRadianGen(), randRadianGen()]
        r2 = [randRadianGen(), randRadianGen(), randRadianGen()]
        r1 = [randRadianGen(), randRadianGen(), randRadianGen()]

        angles = l1 + l2 + l3 + l4 + r4 + r3 + r2 + r1
        population.append(angles)

    return population


def animateTargetChromosomes(title, chrom_list, delay=0.1):
    """Animate a list of target chromosomes using plot_spider_pose.

    Temporarily makes matplotlib's show non-blocking so we can call
    `plot_spider_pose` without modifying that module. Each frame is displayed
    for `delay` seconds and then the figure is closed.
    """
    orig_show = plt.show

    def _non_blocking_show(*args, **kwargs):
        try:
            return orig_show(block=False)
        except TypeError:
            return orig_show(*args, **kwargs)

    plt.show = _non_blocking_show
    try:
        for idx, chrom in enumerate(chrom_list):
            # keep output brief
            if idx % 50 == 0:
                print(f"Animating frame {idx+1}/{len(chrom_list)}")
            plot_spider_pose(chrom)
            # Set a title that shows current frame number
            try:
                ax = plt.gca()
                ax.set_title(f"{title}: Frame {idx+1} out of {len(chrom_list)}")
            except Exception:
                pass
            try:
                # Pause to allow the GUI to update. Do not close the figure here; the
                # plotting function reuses figure 1 and clears it, which produces a
                # smooth animation when we leave the window open.
                plt.pause(delay)
            except Exception:
                # Non-GUI backends may not support pause; ignore
                pass
        # Close all figures at the end of the animation
        try:
            plt.close('all')
        except Exception:
            pass
    finally:
        plt.show = orig_show


# This is our function for calculating the fitness of a chromosome. We do this by taking the target 
# chromosome angles and taking away the generated angles. The closer to 0, the more they overlap, and the more 
# fit they are.

def calculateFitness(inputAngles, targetChromosome):
    diffs = [abs(t - i) for t, i in zip(targetChromosome, inputAngles)]
    fitness = sum(diffs)
    return fitness

# This is a function for calculating the best fitness out of a population. It first sets a bestFitness as 
# infinity so that any fitness calculated will be better. It then loops through the population, calculating the 
# fitness of each chromosome. If the fitness is better than the bestFitness, it updates bestFitness and 
# bestIndex. It also keeps track of the second best fitness and index for when we do crossover later.

def calculateBestFitness(population, targetChromosome):
    bestIndex = 0
    bestFitness = float('inf')
    secondBestFitness = float('inf')
    secondBestIndex = 0
    for i, chrom in enumerate(population):
        currentFitness = calculateFitness(chrom, targetChromosome)
        if currentFitness < bestFitness:
            secondBestFitness = bestFitness
            bestFitness = currentFitness
            secondBestIndex = bestIndex
            bestIndex = i
    return bestIndex, secondBestIndex, bestFitness

# Here we are doing crossover between two chromosomes to create two new ones. We randomly select a crossover 
# point and swap the genes after that point. The best two parents are passed in as arguments.
def crossover(chromosomeA, chromosomeB):
    crossoverPoint = rd.randint(1, len(chromosomeA) - 1)
    newChromosome1 = chromosomeA[:crossoverPoint] + chromosomeB[crossoverPoint:]
    newChromosome2 = chromosomeB[:crossoverPoint] + chromosomeA[crossoverPoint:]
    return newChromosome1, newChromosome2

# This is our mutation function. It goes through each angle in the chromosome and then generate a random number.
# If the random number is less than the mutation rate, it mutates that angle by generating a new random angle.
def mutate(chromosome, mutationRate):
    mutatedChromosome = chromosome.copy()
    for i in range(len(mutatedChromosome)):
        if rd.random() < mutationRate:
            mutatedChromosome[i] = randRadianGen()
    return mutatedChromosome

# We create a new population by doing crossover between the best and second best chromosomes.
# 15 of the new created with the first half of A and second half of B, and 15 with first half of B and 
# second half of A.

def createNewPopulation(best, secondBest, populationSize):
    newPopulation = []
    for i in range(int(populationSize / 2)):
        newChromosomeA, newChromosomeB = crossover(best, secondBest)
        newPopulation.append(newChromosomeA)
        newPopulation.append(newChromosomeB)
    return newPopulation

def runGA(generations, populationSize, mutationRate, GAPoses):

    # The initial two target chromosomes are generated in the main and then
    # passed to here as GAPoses

    # We create the list where the generated frames will be stored
    generatedChromosomeList = []

    for chrom in range(len(GAPoses)):
        population = createRandomPopulation(populationSize)
        targetChromosome = GAPoses[chrom]
        
        bestFitness = float('inf')
        
        for gen in range(generations):
            genBestIndex, secondBestChromosomeIndex, genBestFitness = calculateBestFitness(population, targetChromosome)
            print(f"Frame {chrom + 1}/300 - Generation {gen}: Best Fitness = {100 - genBestFitness}")
            
            # Update global best if this generation found a better one
            if genBestFitness < bestFitness:
                bestFitness = genBestFitness
                bestChromosome = population[genBestIndex]

            newPop = createNewPopulation(population[genBestIndex], population[secondBestChromosomeIndex], populationSize)

            mutatedPop = []
            for i in range(len(newPop)):
                potentiallyMutatedChromosome = mutate(newPop[i], mutationRate)
                mutatedPop.append(potentiallyMutatedChromosome)
            
            population = mutatedPop

        generatedChromosomeList.append(bestChromosome)

    return generatedChromosomeList

