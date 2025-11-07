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
    

def createRandomPopulation():
    populationSize = 30
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

    print(f'Best fitness: {bestFitness} at index {bestIndex}. second best fitness: {secondBestFitness} at index {secondBestIndex}')
    return bestIndex, secondBestIndex

targetChromosone = createTargetChromosone()
population = createRandomPopulation()
print(f'population: \n{population}')
print(f'\ntargetChromosone: {targetChromosone}')
print(f'fitness of population[4]: {calculateFitness(population[4], targetChromosone)}')

