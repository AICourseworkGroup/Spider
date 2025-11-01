import random as rd
import math
from plot_spider_pose import plot_spider_pose

def createTargerChromosone():

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


def randDegGen():
    angle = math.radians(rd.randint(-180, 180))
    return angle
    

def createRandomPopulation():
    populationSize = 30
    population = []

    for i in range(populationSize):
        l1 = [randDegGen(), randDegGen(), randDegGen()]
        l2 = [randDegGen(), randDegGen(), randDegGen()]
        l3 = [randDegGen(), randDegGen(), randDegGen()]
        l4 = [randDegGen(), randDegGen(), randDegGen()]
        r4 = [randDegGen(), randDegGen(), randDegGen()]
        r3 = [randDegGen(), randDegGen(), randDegGen()]
        r2 = [randDegGen(), randDegGen(), randDegGen()]
        r1 = [randDegGen(), randDegGen(), randDegGen()]

        angles = l1 + l2 + l3 + l4 + r4 + r3 + r2 + r1
        population.append(angles)

    return population


targetChromosone = createTargerChromosone()
def calculateFitness(inputAngles, targetChromosone):
    diffs = [abs(t - i) for t, i in zip(targetChromosone, inputAngles)]
    fitness = sum(diffs)
    return fitness

population = createRandomPopulation()
print(f'population: {population}')
print(f'targetChromosone: {targetChromosone}')
print(f'fitness of population[4]: {calculateFitness(population[4], targetChromosone)}')

