# Optimization of the performance of stochastic surveillance strategies
import numpy as np
import math
import random
from StratViz import *

random.seed(1)

def initRandP(A):
    P = np.zeros([A.shape[0], A.shape[1]])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P[i, j] = random.random()
    P = np.matmul(np.diag(1/np.sum(P, axis=1)), P)   # normalize to generate valid prob dist
    return P

A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])



P = initRandP(A)
print(P)

