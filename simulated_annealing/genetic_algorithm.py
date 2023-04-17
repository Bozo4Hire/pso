import math
from random import choices, randint, randrange, random
from typing import Callable, List, Tuple
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optimization_functions as optF
import time

Genome = List[float]
Population = List[Genome]
OptFunction = Callable[[Genome], float]

def generateGenome(length : int, a : float, b: float) -> Genome:
    return [np.random.uniform(a, b) for _ in range(length)]

def generatePopulation(size : int, genomeLength : int, a: float, b: float) -> Population:
    return [generateGenome(genomeLength, a, b) for _ in range(size)]

def simpleMutation(genome: Genome, a : float, b: float) -> Genome:
    i = randrange(len(genome))
    genome[i] = np.random.uniform(a, b)
    return genome

def biasedMutation(genome: Genome, a : float, b: float) -> Genome:
    return

def singlePointCrossover(p1: Genome, p2: Genome) -> Tuple[Genome, Genome]:
    i = randrange(0, len(p1))
    if len(p1) > 1:
        return p1[0:i] + p2[i:], p2[0:i] + p1[i:]
    else:
        raise TypeError("Genome too short to perform crossover function")

def parentSelection(population: Population, optFunc : OptFunction, n: int) -> Population:
    return choices(
        population = population,
        weights = [optFunc(genome) for genome in population],
        k = n
    )

def sortPopulation(population: Population, fitnessFunc: OptFunction) -> Population:
    return sorted(population, key=fitnessFunc, reverse = False)

def geneticAlgorithm(
        popSize : int, 
        genomeLen : int,
        nGenerations : int,
        a : float,
        b : float,
        fitnessFunc : OptFunction) \
        -> Population:
    
    start = time.time()
    gens=np.array([])
    fit =np.array([])
    f = np.array([])
    f_mean =np.array([])

    population = generatePopulation(popSize, genomeLen, a, b)
    population = sortPopulation(population, fitnessFunc)

    for i in range(nGenerations):
        #print(i)

        if fitnessFunc(population[0]) == 0:
            break
        
        newGeneration = population[0:2]
        for j in range(int(len(population) / 2) - 1):
            parents = parentSelection(population, fitnessFunc, 2)
            offspringA, offspringB = singlePointCrossover(parents[0], parents[1])
            offspringA = simpleMutation(offspringA, a, b)
            offspringB = simpleMutation(offspringB, a, b)
            newGeneration += [offspringA, offspringB]

        population = newGeneration
        population = sortPopulation(population, fitnessFunc)
             
        total = 0
        for k in range(0, len(population)-1):
            total += fitnessFunc(population[k])

        gens = np.append(gens, i+1)    
        fit = np.append(fit, fitnessFunc(population[0]))
        f_mean = np.append(f_mean, total/popSize)
    end = time.time()

    print("\nAlgoritmo Genetico")
    print("- Tamaño de Genotipo:\t\t", genomeLen)
    print("- Tamaño de Población:\t\t", popSize)
    print("- Máximo de Generaciones:\t", nGenerations)

    print("- Función de Crossover:\t\t Single Point Crossover")
    print("- Función de Mutación:\t\t Simple Mutation")

    print("\n================Resultados================")
    if fitnessFunc(population[0]) == 0:
        print("Se encontró un óptimo en la generación", i+1)
    else:
        print("No se encontró un óptimo")
    print(f"\tT total de ejecución: {end-start} s")

    print("\nMejor Resultado")
    print("Genotipo:", population[0])
    print("Fitness:", fitnessFunc(population[0]), "\n")

    #Grafica 1
    plt.figure()
    plt.plot(gens,fit)
    plt.title("Algoritmo Genetico")
    plt.xlabel("Generaciones", size = 16,)
    plt.ylabel("Valor maximo de funcion objetivo", size = 12,)
    plt.show()

    #Grafica 2
    plt.figure()
    plt.plot(gens,f_mean)
    plt.title("Algoritmo Genetico")
    plt.xlabel("Generaciones", size = 16)
    plt.ylabel("Fitness promedio", size = 12)
    plt.show()
    
    return population

#newPop = geneticAlgorithm(50, 5, 10000, -5.12, 5.12, optF.rastrigin)
newPop = geneticAlgorithm(50, 2, 10000, -5, 5, optF.threeHumpCamel)
newPop = geneticAlgorithm(50, 2, 10000, -32, 32, optF.rosenbrocksBanana)