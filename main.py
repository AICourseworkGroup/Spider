import random as rd
import math
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

    # Step 2: Divide this difference by 148 to get the amount we need to increment by.
    differenceBetweenChromosones = [d / 148 for d in differenceBetweenChromosones]

    # Here we initialise the list of target chromosones with the initial standing one.
    targetChromosones = [targetChromosoneA]

    # Step 3: For loop that creates the 148 inbetween chromosones. It will do this by adding the difference / 148 to the
    # angles everytime and then appending that target chromosone to the list.
    for i in range(148):
        step = i + 1
        currentTargetChromosone = [a + step * d for a, d in zip(targetChromosoneA, differenceBetweenChromosones)]

        # After calculating the incremented frame we append it to the list before looping again
        targetChromosones.append(currentTargetChromosone)

    # With all intermediary frames added, we can now add the middle (150th) frame
    targetChromosones.append(targetChromosoneB)

    # Step 5: Once we have these initial 150 frames, we can get the final 150 by simply reversing the list and going
    # backwards to return to the starting position

    targetChromosones += targetChromosones[::-1]
    
    #Checking it works

    for i in range(len(targetChromosones)):
        if i % 10 == 0:  # Print and plot every 10th frame for brevity
            print(f"Frame {i}: {targetChromosones[i]}")
            plot_spider_pose(targetChromosones[i])

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

def main(generations, populationSize, mutationRate):

    #We create the two initial target chromosone poses (standing and mid stride)
    targetChromosoneA = createTargetChromosone(math.radians(0), math.radians(-45), math.radians(-30), True)
    targetChromosoneB = createTargetChromosone(math.radians(20), math.radians(-45), math.radians(-30), False)
    createTargetChromosoneList(targetChromosoneA, targetChromosoneB)

    population = createRandomPopulation(populationSize)

    for gen in range(generations):
        bestChromosoneIndex, secondBestChromosoneIndex, bestFitness = calculateBestFitness(population, targetChromosoneA)
        print(f"Generation {gen}: Best Fitness = {100 - bestFitness}")

        newPop = createNewPopulation(population[bestChromosoneIndex], population[secondBestChromosoneIndex], populationSize)

        mutatedPop = []
        for i in range(len(newPop)):
            potentiallyMutatedChromosone = mutate(newPop[i], mutationRate)
            mutatedPop.append(potentiallyMutatedChromosone)
        
        population = mutatedPop

if __name__ == "__main__":
    main(100, 100, 0.1)