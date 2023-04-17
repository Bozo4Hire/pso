import numpy as np
import math

#################### Test functions for single-objective optimization ####################

# global minimum: f(0, ..., 0) = 0
def rastrigin(genome) -> float:
    n = len(genome)
    k=0
    for i in range(0, n):
        if genome[i] > 5.12 or genome[i] < -5.12:
            raise ValueError("rastrigin: Error: value out of range [-5.12, 5,12]")

        k += (genome[i]**2 - 10 * np.cos(2 * np.pi * genome[i]))
    return (10*n) + k

# global minimum: f(0,0) = 0
def threeHumpCamel(genome) -> float:
    if len(genome) != 2:
        raise TypeError("threeHumpCamel: Error: invalid input type")
    
    x = genome[0]
    y = genome[1]

    if x > 5 or x < -5 or y > 5 or y < -5:
         raise ValueError("threeHumpCamel: Error: value out of range [-5, 5]")
    
    return 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2

# global minimum: f(0,0) = 0
def ackley(genome) -> float:
    if len(genome) != 2:
        raise TypeError("ackley: Error: invalid input type")
    
    x = genome[0]
    y = genome[1]

    if x > 5 or x < -5 or y > 5 or y < -5:
         raise ValueError("ackley: Error: value out of range [-5, 5]")
    
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

# global minumum: f(0, ..., 0) = 0
def sphere(genome) -> float:
    return sum([(i**2) for i in genome])

# global minumum: f(1, ..., 1) = 0
def rosenbrocksBanana(genome) -> float:
    if len(genome) < 2:
        raise TypeError("rosenbrocksBanana: Error: invalid input type")
    
    k = 0 
    for i in range(0, len(genome)-1):
        k += 100 * (genome[i+1] - genome[i]**2)**2 + (1 - genome[i]**2)
    return k 