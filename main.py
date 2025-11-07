import random as rd
import math
from plot_spider_pose import plot_spider_pose

def createTargetChromosone():

    a = math.radians(0)
    b = math.radians(-45)
    c = math.radians(-30)

    l1 = [a, b, c]
    l2 = [a, b, c]
    l3 = [a, b, c]
    l4 = [a, b, c]
    r4 = [-a, b, c]
    r3 = [-a, b, c]
    r2 = [-a, b, c]
    r1 = [-a, b, c]

    # return a flat Python list (24 floats) in radians
    angles = l1 + l2 + l3 + l4 + r4 + r3 + r2 + r1
    return angles


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
    targetChromosone = createTargetChromosone()
    population = createRandomPopulation(populationSize)

    for gen in range(generations):
        bestChromosoneIndex, secondBestChromosoneIndex, bestFitness = calculateBestFitness(population, targetChromosone)
        print(f"Generation {gen}: Best Fitness = {bestFitness}")

        newPop = createNewPopulation(population[bestChromosoneIndex], population[secondBestChromosoneIndex], populationSize)

        mutatedPop = []
        for i in range(len(newPop)):
            potentiallyMutatedChromosone = mutate(newPop[i], mutationRate)
            mutatedPop.append(potentiallyMutatedChromosone)
        
        population = mutatedPop

if __name__ == "__main__":
    main(100, 100, 0.1)