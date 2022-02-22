# Optimization of the performance of stochastic surveillance strategies
import numpy as np
np.set_printoptions(linewidth=np.inf)
import math
from StratComp import *

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


# Apply constraints to gradient ascent steps suggested by gradients of min cap probs,
# compare the min cap probs which result from taking the allowed steps, select the
# best step and return the corresponding updated transition probability matrix P
def gradAscentStep(P, J, F, tau, eps):
    n = P.shape[0]
    Fvec = F.flatten('F')
    # get indices of minimal capture probabilities in cap prob vector
    minCapProb = np.min(Fvec)
    mcpLocs = np.argwhere(Fvec == minCapProb)
    mcpNum = mcpLocs.shape[0]
    # compute corresponding unconstrained gradient ascent steps
    uGradSteps = np.zeros([n**2, mcpNum])
    for i in range(mcpNum):
        uGradSteps[:, i] = eps*J[mcpLocs[i], :].reshape(n**2)
    # compare the min cap probs resulting from each of the above grad steps
    oldPvec = P.flatten('F')
    # take each of the constrained gradient steps under consideration:
    newPvecs = np.outer(oldPvec, np.ones([1, mcpNum])) + uGradSteps
    # reshape each Pvec into a P matrix so that we can compute cap probs:
    newPmats = np.full([n, n, mcpNum], np.nan)  
    minCapProbs = np.zeros(mcpNum)
    capProbs = np.full([n, n, mcpNum], np.nan)
    # compute P matrix, corresponding cap prob mat, and min cap prob for each potential new Pvec
    for k in range(mcpNum):
        newPmat = newPvecs[:, k].reshape((n, n), order='F')
        newPmat = projOntoSimplex(newPmat)
        newPmats[:, :, k] = newPmat
        capProbs[:, :, k]  = computeCapProbs(newPmats[:, :, k], tau)
        minCapProbs[k] = np.min(capProbs[:, :, k])
    # find P matrices which give maximum new min cap probs
    maxMinIndices = np.argwhere(minCapProbs == np.max(minCapProbs))
    newPF = np.full([n, n, 2], np.nan)
    # return new P matrix and cap prob mat corresponding to (an) optimal step choice
    newPF[:, :, 0] = newPmats[:, :, maxMinIndices[0]].reshape(n, n)
    newPF[:, : , 1] = capProbs[:, :, maxMinIndices[0]].reshape(n, n)
    return newPF

# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscentFixed(P0, A, tau, eps):
    iterations = 1001
    P = P0
    F = computeCapProbs(P, tau)
    for k in range(iterations):
        J = compCPJac(P, tau)
        J = zeroCPJacCols(J, A)
        newPF = gradAscentStep(P, J, F, tau, eps)
        P = newPF[:, :, 0]
        F = newPF[:, :, 1]
        if k % ((iterations - 1)/10) == 0:
            print("Minimum Capture Probability at iteration " + str(k) + ":")
            print(np.min(F))
            # print("P at iteration " + str(k) + ":")
            # print(P)
            # print("F at iteration " + str(k) + ":")
            # print(F)
    return P, F


# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscent(P0, A, tau, eps0, radius):
    epsThresh = 0.001  # threshold (i.e., minimum) step size to be used
    thresh = 1000  # step number at which step size reaches threshold value
    iter = 0  # number of gradient ascent steps taken so far
    P = P0 
    F = computeCapProbs(P, tau)
    avgF= F  # running avg capture probability matrix, for checking convergence
    converged = False
    while not converged:
        # set step size
        if iter <= thresh:
            eps = (1 - iter/thresh)*eps0 + (iter/thresh)*epsThresh
        else:
            eps = epsThresh
        # take gradient ascent step:
        J = compCPJac(P, tau)
        J = zeroCPJacCols(J, A)
        newPF = gradAscentStep(P, J, F, tau, eps)
        P = newPF[:, :, 0]
        F = newPF[:, :, 1]
        # print status info to terminal:
        if (iter % 50) == 0:
            print("Minimum Capture Probability at iteration " + str(iter) + ":")
            print(np.min(F))
            # print("F at iteration " + str(iter) + ":")
            # print(F)
        # check for convergence, update running avg cap probs and step counter:
        newAvgF = ((iter)*avgF + F)/(iter + 1)
        diffAvgF = np.abs(newAvgF - avgF)
        avgF = newAvgF
        converged = np.amax(diffAvgF) < radius
        iter = iter + 1
    print("Minimum Capture Probability at iteration " + str(iter - 1) + ":")
    print(np.min(F))
    print("Final diffAvgF = ")
    print(diffAvgF)
    return P, F

# TESTING ------------------------------------------------------------------------
# A = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
# A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
# A = np.array([[0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
P0 = initRandP(A)
# P0 = np.array([[0, 0.2, 1 - 0.2], [1, 0, 0], [1, 0, 0]])
tau = 3

np.set_printoptions(suppress=True)
# [P, F] = gradAscentFixed(P0, A, tau, 0.05)
[P, F] = gradAscent(P0, A, tau, 0.05, 0.0005)
print("P0 = ")
print(P0)
print("P_final = ")
print(P)

# print("F_initial = ")
F0 = computeCapProbs(P0, tau)
# print(F0)
# print("F_final = ")
# print(F)

Pdiff = P - P0
Fdiff = F - F0
print("P_diff = ")
print(Pdiff)
print("F_diff = ")
print(Fdiff)

