import random as rd
import math
import matplotlib.pyplot as plt
from plot_spider_pose import plot_spider_pose

def createTargetChromosome(a, b, c, isA):

    # isA checks whether the target chromosome being made is for the standing pose or the mid walk pose and sets the values
    # of the a angles to positive or negative accordingly.
    if isA:
        l1 = [a, b, c]
        l2 = [a, b, c]
        l3 = [a, b, c]
        l4 = [a, b, c]
        r4 = [-a, b, c]
        r3 = [-a, b, c]
        r2 = [-a, b, c]
        r1 = [-a, b, c]
    else:
        l1 = [a, b, c]
        l2 = [-a, b, c]
        l3 = [a, b, c]
        l4 = [-a, b, c]
        r4 = [a, b, c]
        r3 = [-a, b, c]
        r2 = [a, b, c]
        r1 = [-a, b, c]


    # return a flat Python list (24 floats) in radians
    angles = l1 + l2 + l3 + l4 + r4 + r3 + r2 + r1

    print(f"Target chromosome: {angles} ") 
    #plot_spider_pose(angles)

    if isA:
        title = "Target Chromosome A"
        plot_spider_pose(angles, title=title)
    else:
        title = "Target Chromosome B"
        plot_spider_pose(angles, title=title)

    return angles

def createTargetChromosomeList(targetChromosomeA, targetChromosomeB):

    # Step 1: Take away the difference between the two chromosomes.
    differenceBetweenChromosomes = [b - a for a, b in zip(targetChromosomeA, targetChromosomeB)]

    # Step 2: Divide this difference by 149 to get the amount we need to increment by.
    differenceBetweenChromosomes = [d / 149 for d in differenceBetweenChromosomes]

    # Here we initialise the list of target chromosomes with the initial standing one.
    targetChromosomes = [targetChromosomeA]

    # Step 3: For loop that creates the 149 inbetween chromosomes. It will do this by adding the difference / 149 to the
    # angles everytime and then appending that target chromosome to the list.
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
    
    # At the end of creating the full list, run an animation so you can inspect
    # the full gait sequence. This calls the local helper which temporarily
    # makes matplotlib non-blocking. If the environment doesn't support GUI
    # display this will quietly fail.
    #try:
        #animate_target_chromosomes(targetChromosomes, delay=0.1)
    #except Exception:
        # Ignore animation errors so creation still returns the list
        #pass


    # This is a list of every target chromosome for all 300 frames. Every time we generate a new target we append it to 
    # this list. When we run the genetic algorithm we'll do a for loop where we go to the next target chromosome every time
    # that i increments.
    
    return targetChromosomes

def randRadianGen():
    angle = math.radians(rd.randint(-180, 180))
    return angle

def createRandomPopulation(populationSize = 30):
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


def animateTargetChromosomes(chrom_list, delay=0.1):
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
                ax.set_title(f"Frame {idx+1} out of {len(chrom_list)}")
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

def calculateFitness(inputAngles, targetChromosome):
    diffs = [abs(t - i) for t, i in zip(targetChromosome, inputAngles)]
    fitness = sum(diffs)
    return fitness

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


def crossover(chromosomeA, chromosomeB):
    crossoverPoint = rd.randint(1, len(chromosomeA) - 1)
    newChromosome1 = chromosomeA[:crossoverPoint] + chromosomeB[crossoverPoint:]
    newChromosome2 = chromosomeB[:crossoverPoint] + chromosomeA[crossoverPoint:]
    return newChromosome1, newChromosome2

def mutate(chromosome, mutationRate):
    mutatedChromosome = chromosome.copy()
    for i in range(len(mutatedChromosome)):
        if rd.random() < mutationRate:
            mutatedChromosome[i] = randRadianGen()
    return mutatedChromosome

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
            print(f"Generation {gen}: Best Fitness = {100 - genBestFitness}")
            
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
        # Plot the best chromosome from ALL generations for this target
        #title = f"Best Fitness: {100 - bestFitness}. Frame number: {chrom + 1} out of 300"
        #plot_spider_pose(bestChromosome, title=title)

        generatedChromosomeList.append(bestChromosome)

    #Animate
    try:
        animateTargetChromosomes(generatedChromosomeList, delay=0.1)
    except Exception:
        #Ignore animation errors so creation still returns the list
        pass
    
    return generatedChromosomeList

if __name__ == "__main__":
    runGA(100, 300, 0.1)

