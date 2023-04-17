import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple
import simulated_annealing.optimization_functions as optF
import time

VelVector = List[float] 
PosVector = List[float]
PointCloud = Tuple [List[PosVector], List[VelVector]]

OptFunction = Callable[[PosVector], float]
VelFunction = Callable[[PosVector, VelVector, float, float, PosVector, PosVector], VelVector]

def generatePointCloud(cloudSize : int, vectorSize : float, a: int, b: int) -> PointCloud:
    return  [generateVector(vectorSize, a, b) for _ in range(cloudSize)], \
            [generateVector(vectorSize, 0, 0.1) for _ in range(cloudSize)]

def generateVector(size : int, a: int, b: int):
    return [np.random.uniform(a, b) for _ in range(size)]


def og_V_Update_Func(pVector: PosVector, vVector: VelVector, phi1: float, phi2 : float,  p: PosVector, g: PosVector) -> VelVector:
    return np.add(vVector, np.add(np.multiply(np.random.uniform(0, phi1), np.subtract(p, pVector)), np.multiply(np.random.uniform(0, phi2), np.subtract(g, pVector))))

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
    
    cloud = generatePointCloud(cloudSize,vectorSize,a,b)
    p = [a,b] 
    g = cloud[0][0] 
    for z in range(0, maxit):
        costs = [objectiveFunc(_) for _ in cloud[0]]
        gbest = objectiveFunc(g)
        pbest = objectiveFunc(p)

        for i in range(0, len(cloud[0])):
            if costs[i] < pbest:
                p = cloud[0][i]
            if costs[i] < gbest:
                g = cloud[0][i]

        for i in range(0, len(cloud[0])):
            cloud[1][i] = velFuction(cloud[0][i], cloud[1][i], phi1, phi2, p, g)
            cloud[0][i] = np.add(cloud[0][i], cloud[1][i]) # duda importante sobre superar los limites de la funcion
        
        print("Best:", p, " ", objectiveFunc(p))
    return PointCloud

pso(optF.ackley, og_V_Update_Func, 20, 2, 0.1, 0.1, -5, 5, 100)
