# Computation of quantities relevant to optimization of stochastic surveillance strategies
import numpy as np
import math
import random

# Takes:   A, the binary adjacency matrix corresponding to the environment graph
# Returns: P0, a random transition probability matrix which is valid (i.e. row-stochastic) 
#              and consistent with the environment graph described by A (see XD-DP-FB_19b.pdf)
def initRandP(A):
    random.seed(2)
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

# Computes Jacobians of the first hitting time probability matrices F_k
def compFkJacs(P, tau):
    n = P.shape[0]  # get number of nodes
    F = computeFHTProbMats(P, tau) # get first hitting time probability matrices
    J = np.full([n**2, n**2, tau], np.NaN) # initialize array to store Jacobians of FHT probability matrices
    J[:, :, 0] = np.identity(n**2) # Jacobian of F_1 is the identity matrix
    # generate matrices needed for Jacobian computations :
    Pbar = np.zeros([n**2, n**2])
    for i in range(n):
        Pbar[i*n:(i + 1)*n, i*(n + 1)] = P[:, i]
    B = np.kron(np.identity(n), P) - Pbar
    # recursive computating of Jacobians of FHT probability matrices:
    for i in range(1, tau):
        J[:, :, i] = np.kron(np.transpose(F[:, :, i - 1]), np.identity(n)) - np.kron(np.diag(np.diag(F[:, :, i - 1])), np.identity(n)) + np.matmul(B, J[:, :, i - 1])

    return J

# Sums Jacobians of FHT probability matrices to get Jacobian of Capture Probability Matrix
def compCPJac(P, tau):
    J = compFkJacs(P, tau)
    CPGrad = np.sum(J, axis=2)
    return CPGrad

# Zero's-out the columns of the Jacobian of the Capture Probability Matrix corresponding
# to edges which are not in the environment graph 
# Takes:   Jacobian of the Capture Probability Matrix, Binary Adjacency Matrix for env graph
# Returns: Jacobian with appropriate columns set to zero
def zeroCPJacCols(J, A):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i, j] == 0:
                J[:, i*n + j] = np.zeros([n**2])
    return J

# Project the given trans prob vector onto nearest point on probability simplex, if applicable
# See https://arxiv.org/abs/1309.1541 for explanation of the algorithm used here
def projOntoSimplex(P):
    n = P.shape[0]
    sortMapping = np.fliplr(np.argsort(P, axis=1))
    X = np.full_like(P, np.nan)
    for i  in range (n):
        for j in range (n):
            X[i, j] = P[i, sortMapping[i, j]]
    Xtmp = np.matmul(np.cumsum(X, axis=1) - 1, np.diag(1/np.arange(1, n + 1)))
    rhoVals = np.sum(X > Xtmp, axis=1) - 1
    lambdaVals = -Xtmp[np.arange(n), rhoVals]
    newX = np.maximum(X + np.outer(lambdaVals, np.ones(n)), np.zeros([n, n]))
    newP = np.full_like(P, np.nan)
    for i in range(n):
        for j in range(n):
            newP[i, sortMapping[i, j]] = newX[i, j]
    return newP

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
