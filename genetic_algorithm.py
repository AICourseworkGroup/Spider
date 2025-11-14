import random as rd
import math
import matplotlib.pyplot as plt
from plot_spider_pose import plot_spider_pose

def createTargetChromosone(a, b, c, isA):

    # isA checks whether the target chromosone being made is for the standing pose or the mid walk pose and sets the values
    # of the a angles to positive or negative accordingly.
    if isA == True:
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

    print(f"Target chromosone: {angles} ") 
    plot_spider_pose(angles)

    return angles

def createTargetChromosoneList(targetChromosoneA, targetChromosoneB):

    # Step 1: Take away the difference between the two chromosones.
    differenceBetweenChromosones = [b - a for a, b in zip(targetChromosoneA, targetChromosoneB)]

    # Step 2: Divide this difference by 149 to get the amount we need to increment by.
    differenceBetweenChromosones = [d / 149 for d in differenceBetweenChromosones]

    # Here we initialise the list of target chromosones with the initial standing one.
    targetChromosones = [targetChromosoneA]

    # Step 3: For loop that creates the 149 inbetween chromosones. It will do this by adding the difference / 149 to the
    # angles everytime and then appending that target chromosone to the list.
    for i in range(149):
        step = i + 1
        currentTargetChromosone = [a + step * d for a, d in zip(targetChromosoneA, differenceBetweenChromosones)]

        # After calculating the incremented frame we append it to the list before looping again
        targetChromosones.append(currentTargetChromosone)

    # Step 4: We will now take away instead of adding to complete the 2nd half of the walk cycle
    for i in range(149):
        step = i + 1
        currentTargetChromosone = [a - step * d for a, d in zip(targetChromosoneB, differenceBetweenChromosones)]

        # After calculating the incremented frame we append it to the list before looping again
        targetChromosones.append(currentTargetChromosone)

    # With all intermediary frames added, we can now add the final (300th) frame
    targetChromosones.append(targetChromosoneA)

    #Checking it works
    #for i in range(len(targetChromosones)):
        #if i % 50 == 0:  # Print and plot every 10th frame for brevity
            #print(f"Frame {i}: {targetChromosones[i]}")
            #plot_spider_pose(targetChromosones[i])
    
    # At the end of creating the full list, run an animation so you can inspect
    # the full gait sequence. This calls the local helper which temporarily
    # makes matplotlib non-blocking. If the environment doesn't support GUI
    # display this will quietly fail.
    #try:
        #animate_target_chromosomes(targetChromosones, delay=0.1)
    #except Exception:
        # Ignore animation errors so creation still returns the list
        #pass


    # This is a list of every target chromosone for all 300 frames. Every time we generate a new target we append it to 
    # this list. When we run the genetic algorithm we'll do a for loop where we go to the next target chromosone every time
    # that i increments.
    
    return targetChromosones

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


def animate_target_chromosomes(chrom_list, delay=0.1):
    """Animate a list of target chromosones using plot_spider_pose.

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

def calculateFitness(inputAngles, targetChromosone):
    diffs = [abs(t - i) for t, i in zip(targetChromosone, inputAngles)]
    fitness = sum(diffs)
    return fitness

def calculateBestFitness(population, targetChromosone):
    bestIndex = 0
    bestFitness = float('inf')
    secondBestFitness = float('inf')
    secondBestIndex = None
    for i, chrom in enumerate(population):
        currentFitness = calculateFitness(chrom, targetChromosone)
        if currentFitness < bestFitness:
            secondBestFitness = bestFitness
            bestFitness = currentFitness
            secondBestIndex = bestIndex
            bestIndex = i
    return bestIndex, secondBestIndex, bestFitness


def crossover(chromasoneA, chromasoneB):
    crossoverPoint = rd.randint(1, len(chromasoneA) - 1)
    newChromosone1 = chromasoneA[:crossoverPoint] + chromasoneB[crossoverPoint:]
    newChromosone2 = chromasoneB[:crossoverPoint] + chromasoneA[crossoverPoint:]
    return newChromosone1, newChromosone2

def mutate(chromosone, mutationRate):
    mutatedChromosone = chromosone.copy()
    for i in range(len(mutatedChromosone)):
        if rd.random() < mutationRate:
            mutatedChromosone[i] = randRadianGen()
    return mutatedChromosone

def createNewPopulation(best, secondBest, populationSize):
    newPopulation = []
    for i in range(int(populationSize / 2)):
        newChromosoneA, newChromosoneB = crossover(best, secondBest)
        newPopulation.append(newChromosoneA)
        newPopulation.append(newChromosoneB)
    return newPopulation

def run_ga(generations, populationSize, mutationRate):

    #We create the two initial target chromosone poses (standing and mid stride)
    targetChromosoneA = createTargetChromosone(math.radians(0), math.radians(-45), math.radians(-30), True)
    targetChromosoneB = createTargetChromosone(math.radians(20), math.radians(-45), math.radians(-30), False)
    targetChromosoneList = createTargetChromosoneList(targetChromosoneA, targetChromosoneB)

    # We create the list where the generated frames will be stored
    generatedChromosoneList = []

    for chrom in range(len(targetChromosoneList)):
        population = createRandomPopulation(populationSize)
        targetChromosone = targetChromosoneList[chrom]
        
        # Track the best across all generations for this frame
        bestChromosoneIndex = 0
        bestFitness = float('inf')
        bestChromosone = population[0]
        
        for gen in range(generations):
            genBestIndex, secondBestChromosoneIndex, genBestFitness = calculateBestFitness(population, targetChromosone)
            print(f"Generation {gen}: Best Fitness = {100 - genBestFitness}")
            
            # Update global best if this generation found a better one
            if genBestFitness < bestFitness:
                bestFitness = genBestFitness
                bestChromosoneIndex = genBestIndex
                bestChromosone = population[genBestIndex]

            newPop = createNewPopulation(population[genBestIndex], population[secondBestChromosoneIndex], populationSize)

            mutatedPop = []
            for i in range(len(newPop)):
                potentiallyMutatedChromosone = mutate(newPop[i], mutationRate)
                mutatedPop.append(potentiallyMutatedChromosone)
            
            population = mutatedPop
        # Plot the best chromosone from ALL generations for this target
        #title = f"Best Fitness: {100 - bestFitness}. Frame number: {chrom + 1} out of 300"
        #plot_spider_pose(bestChromosone, title=title)

        generatedChromosoneList.append(bestChromosone)

    #Animate
    try:
        animate_target_chromosomes(generatedChromosoneList, delay=0.1)
    except Exception:
        #Ignore animation errors so creation still returns the list
        pass
    
    return generatedChromosoneList

if __name__ == "__main__":
    run_ga(300, 300, 0.1)
