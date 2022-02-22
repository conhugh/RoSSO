# Computation of quantities relevant to stochastic surveillance strategies
import numpy as np
import math
import random

# Takes:   A, the binary adjacency matrix corresponding to the environment graph
# Returns: P0, a random transition probability matrix which is valid (i.e. row-stochastic) 
#              and consistent with the environment graph described by A (see XD-DP-FB_19b.pdf)
def initRandP(A):
    # random.seed(1)
    P0 = np.zeros([A.shape[0], A.shape[1]])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P0[i, j] = random.random()
    P0 = np.matmul(np.diag(1/np.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0
    
# Compute first hitting time probability matrices (up to tau time steps) for fixed P matrix 
def computeFHTProbMats(P, tau):
    # print("tau = " + str(tau))
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

# Print the given set of first hitting time probability matrices 
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
