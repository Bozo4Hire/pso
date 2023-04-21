import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple
import simulated_annealing.optimization_functions as optF
import time

VelVector = List[float] 
PosVector = List[float]
PointCloud = Tuple [List[PosVector], List[VelVector], List[PosVector], List[PosVector]]

OptFunction = Callable[[PosVector], float]
VelFunction = Callable[[PosVector, VelVector, float, float, PosVector, PosVector], VelVector]

def generatePointCloud(cloudSize : int, vectorSize : float, a: int, b: int) -> PointCloud:
    return  [generateVector(vectorSize, a, b) for _ in range(cloudSize)], \
            [generateVector(vectorSize, 0, 0.1) for _ in range(cloudSize)], \
            [generateVector(vectorSize, 0, 0) for _ in range(cloudSize)], \
            [generateVector(vectorSize, a, a) for _ in range(cloudSize)]

def generateVector(size : int, a: int, b: int):
    return [np.random.uniform(a, b) for _ in range(size)]

def og_V_Update_Func(pVector: PosVector, vVector: VelVector, phi1: float, phi2 : float,  p: PosVector, g: PosVector, a:int, b:int) -> VelVector:
    val = np.add(vVector, np.add(np.multiply(np.random.uniform(0, phi1), np.subtract(p, pVector)), np.multiply(np.random.uniform(0, phi2), np.subtract(g, pVector))))
    return val

def pso(objectiveFunc: OptFunction,
        velFuction: VelFunction,
        cloudSize : int, 
        vectorSize : int, 
        phi1: float,
        phi2: float,
        a: int, 
        b: int,
        maxit: int) \
            -> PointCloud:
    
    iterations = np.array([])
    bestSolutionValue = np.array([])

    k = 5
    cloud = generatePointCloud(cloudSize,vectorSize,a,b)
    c = [a,b] 
    cbest = objectiveFunc(c)

    for z in range(0, maxit):
        costs = [objectiveFunc(_) for _ in cloud[0]]

        for i in range(0, len(cloud[0])):
            # actualizamos mejor resultado
            if costs[i] < cbest:
                c = cloud[0][i]

            if costs[i] < objectiveFunc(cloud[3][i]):
                cloud[3][i] = cloud[0][i]

            # obtenemos un set aleatorio de vecinos para la particula i
            aux = []
            while len(aux) < k: 
                x = random.sample(range(0,len(cloud[0])-1),1)
                if x != i:
                    aux.append(x)

            aux.append(i)
            cloud[2][i] = cloud[0][aux[0][0]]

            # localizamos al mejor vecino
            for j in range(1, len(aux)-1):
                x = aux[j][0]
                if objectiveFunc(cloud[0][x]) < objectiveFunc(cloud[2][i]):
                     cloud[2][i] = cloud[0][x]

        for i in range(0, len(cloud[0])):
            # actualizamos la funcion de velocidad y la posicion de cada particula 
            cloud[1][i] = velFuction(cloud[0][i], cloud[1][i], phi1, phi2, cloud[3][i], cloud[2][i], a, b)
            cloud[0][i] = np.add(cloud[0][i], cloud[1][i])
            
            # verificamos que las posiciones se encuentren dentro del intervalo delimitado
            for j in range (0, len(cloud[0][i])):
                if cloud[0][i][j] < a:
                    cloud[0][i][j] = a
                if cloud[0][i][j] > b:
                    cloud[0][i][j] = b

        cbest = objectiveFunc(c)
        #print("Best:", c, " ", cbest)
        iterations = np.append(iterations, z+1)
        bestSolutionValue = np.append(bestSolutionValue, cbest)
    
    print("Best:", c, " ", cbest)
    # Grafica de Mejor Soluciones por iteracion
    plt.figure()
    plt.plot(iterations, bestSolutionValue)
    plt.title("PSO")
    plt.xlabel("Número de Iteración")
    plt.ylabel("E(S)")
    plt.show()
    
    return cloud

pso(optF.rastrigin, og_V_Update_Func, 50, 2, 0.3, 0.8, -5.12, 5.12, 10000)
pso(optF.threeHumpCamel, og_V_Update_Func, 50, 2, 0.3, 0.8, -5, 5, 10000)
pso(optF.rosenbrocksBanana, og_V_Update_Func, 50, 2, 0.3, 0.8, -32, 32, 10000)