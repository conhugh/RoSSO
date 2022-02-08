# Computation of quantities relevant to stochastic surveillance strategies

import numpy as np
import math
import random

random.seed(1)

def initRandP(A):
    P = np.zeros([A.shape[0], A.shape[1]])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P[i, j] = random.random()
    P = np.matmul(np.diag(1/np.sum(P, axis=1)), P)   # normalize to generate valid prob dist
    return P
    

# Compute first hitting time probability matrices for fixed P matrix
def computeFHTProbMats(P, tau):
    F = np.full([P.shape[0], P.shape[1], tau], np.NaN)
    F[:, :, 0] = P
    for i in range(1, tau):
        F[:, :, i] = np.matmul(P, (F[:, :, i - 1] - np.diag(np.diag(F[:, :, i - 1]))))
    return F


# Compute capture probabilities for each pair of nodes for fixed P matrix
def computeCapProbs(P, tau):
    F = computeFHTProbMats(P, tau)
    capProbs = np.sum(F, axis=2)
    return capProbs

def printFHTProbMats(F):
    for i in range(F.shape[2]):
        print("F_" + str(i + 1) + " = " + "\n" + str(F[:, :, i]))

# TESTING -----------------------------------------------------------------------------------------

# P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # initialize transition prob matrix
# # P = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # initialize transition prob matrix
# P = np.matmul(np.diag(1/np.sum(P, axis=1)), P)   # normalize to generate valid prob dist

# tau = 3  # attack duration

# fhtProbs = computeFHTProbMats(P, tau)
# capProbs = computeCapProbs(P, tau)

# minCapProb = np.min(capProbs)
# mcpLocs = np.argwhere(capProbs == minCapProb)

# print("First Hitting Time Probability Matrices: ")
# printFHTProbMats(fhtProbs)

# print("Capture Probabilities: ")
# print(capProbs)

# A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
# P = initRandP(A)
# print(P)
