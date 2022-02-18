# Optimization of the performance of stochastic surveillance strategies

import numpy as np
np.set_printoptions(linewidth=np.inf)
import math
from StratComp import *

# Computes Jacobians of the first hitting time probability matrices F_k
def compFkJacs(P, tau):
    n = P.shape[0]  # get number of nodes
    F = computeFHTProbMats(P, tau) # get first hitting time probability matrices
    J = np.full([n**2, n**2, tau], np.NaN) # initialize array to store gradients of FHT probability matrices
    J[:, :, 0] = np.identity(n**2) # gradient of F_1 is the identity matrix

    # initialize matrices needed for gradient computations:
    Pbar = np.zeros([n**2, n**2])
    for i in range(n):
        Pbar[i*n:(i + 1)*n, i*(n + 1)] = P[:, i]

    B = np.kron(np.identity(n), P) - Pbar

    # file = open("variables.txt", "w")
    # file.write("B = \n")
    # file.write(np.array2string(B) + "\n")
    # file.write("J_0 = \n")
    # file.write(np.array2string(J[:, :, 0]) + "\n")

    # recursive computating of gradients of FHT probability matrices:
    for i in range(1, tau):
        J[:, :, i] = np.kron(np.transpose(F[:, :, i - 1]), np.identity(n)) - np.kron(np.diag(np.diag(F[:, :, i - 1])), np.identity(n)) + np.matmul(B, J[:, :, i - 1])

        # file.write("F_" + str(i) + " transpose kron I = \n")
        # file.write(np.array2string(np.kron(np.transpose(F[:, :, i - 1]), np.identity(n))) + "\n")
        # file.write("diag(F_" + str(i) + ") kron I = \n")
        # file.write(np.array2string(np.kron(np.diag(np.diag(F[:, :, i - 1])), np.identity(n))) + "\n")
        # file.write("A_" + str(i) + " = \n")
        # file.write(np.array2string(np.kron(np.transpose(F[:, :, i - 1]), np.identity(n)) - np.kron(np.diag(np.diag(F[:, :, i - 1])), np.identity(n))) + "\n")
        # file.write("J_" + str(i) + " = \n")
        # file.write(np.array2string(J[:, :, i]) + "\n")
    # file.close()
    return J


# Sums Jacobians of FHT probability matrices to get Jacobian of Capture Probability Matrix
def compCPJac(P, tau):
    J = compFkJacs(P, tau)
    CPGrad = np.sum(J, axis=2)
    return CPGrad


# Zero's-out the columns of the Jacobian of the Capture Probability Matrix corresponding
# to edges which are not in the environment graph 
# Takes: Jacobian of the Capture Probability Matrix, Binary Adjacency Matrix for env graph
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
# best one and return the updated transition probability matrix P
def gradAscentStep(P, J, F, eps, tau):
    n = P.shape[0]
    # get indices of minimal capture probabilities in cap prob vector
    minCapProb = np.min(F)
    Fvec = F.flatten('F')
    mcpLocs = np.argwhere(Fvec == minCapProb)
    # compute corresponding unconstrained gradient ascent steps
    uGradSteps = np.zeros([n**2, mcpLocs.shape[0]])
    for i in range(mcpLocs.shape[0]):
        uGradSteps[:, i] = eps*np.transpose(J[mcpLocs[i], :]).reshape(n**2)

    # project each onto the subspace of zero-sum n^2-dimensional vectors 
    # [^^ REMOVED FOR NOW, THINK MORE ABOUT THIS]

    # compare the min cap probs resulting from each of the above grad steps
    oldPvec = P.flatten('F')
    mcpOpts = np.full_like(uGradSteps, np.nan)
    # take each of the constrained gradient steps under consideration:
    newPvecs = np.outer(oldPvec, np.ones([1, mcpLocs.shape[0]])) + uGradSteps
    # reshape each Pvec into a P matrix so that we can compute cap probs:
    newPmats = np.zeros([n, n, mcpLocs.shape[0]])  # [LOOK FOR A WAY TO COMPUTE DIRECTLY IN VEC FORM WITHOUT RESHAPING]
    newPmats[:] = np.nan
    minCapProbs = np.zeros(mcpLocs.shape[0])
    capProbs = np.zeros([n, n, mcpLocs.shape[0]])
    capProbs[:] = np.nan
    # compute P matrix, corresponding cap prob mat, min cap prob for each potential new Pvec
    for k in range(mcpLocs.shape[0]):
        newPmat = newPvecs[:, k].reshape((n, n), order='F')
        newPmat = projOntoSimplex(newPmat)
        newPmats[:, :, k] = newPmat
        capProbs[:, :, k]  = computeCapProbs(newPmats[:, :, k], tau)
        minCapProbs[k] = np.min(capProbs[:, :, k])
    # find P matrices which give maximum new min cap probs
    maxMinIndices = np.argwhere(minCapProbs == np.max(minCapProbs))
    newPF = np.zeros((n, n, 2))
    newPF[:] = np.nan
    # return new P matrix and cap prob mat corresponding to (an) optimal step choice
    newPF[:, :, 0] = newPmats[:, :, maxMinIndices[0]].reshape(n, n)
    newPF[:, : , 1] = capProbs[:, :, maxMinIndices[0]].reshape(n, n)
    return newPF

# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscent(P0, A, tau):
    iterations = 1001
    eps = 0.05
    P = P0
    F = computeCapProbs(P, tau)
    for k in range(iterations):
        J = compCPJac(P, tau)
        J = zeroCPJacCols(J, A)
        newPF = gradAscentStep(P, J, F, eps, tau)
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


# TESTING ------------------------------------------------------------------------
# A = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
# A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
# A = np.array([[0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
P0 = initRandP(A)
# P0 = np.array([[0, 0.2, 1 - 0.2], [1, 0, 0], [1, 0, 0]])
tau = 3

np.set_printoptions(suppress=True)
[P, F] = gradAscent(P0, A, tau)
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


# print("P0 = ")
# print(P0)
# np.random.seed(1)
# Ptilde = initRandP(A)
# Ptilde = Ptilde - 0.5
# print("Ptilde = ")
# print(Ptilde)
# zsPtilde = Ptilde - (1/3)*np.matmul(np.ones([3, 3]), Ptilde)
# print("zsPtilde = ")
# print(zsPtilde)
# P1 = P0 + zsPtilde
# print("P0 + zsPtilde = P1 = ")
# print(P1)
# print("P1 row sums = ")
# print(np.sum(P1, axis=1))

# Pproj = projOntoSimplex(P1)
# print("Pproj = ")
# print(Pproj)

# print("Pproj row sums = ")
# print(np.sum(Pproj, axis=1))

# sortMapping = np.fliplr(np.argsort(P0, axis=1))
# print(sortMapping)
# U = np.zeros_like(P0)
# for i  in range (P0.shape[0]):
#     for j in range (P0.shape[1]):
#         U[i, j] = P0[i, sortMapping[i, j]]
# print(U)
# Usums = np.cumsum(U, axis=1)
# print(Usums)

# c = np.zeros_like(P0)
# c[:] = 0.95
# compa = Usums > c
# print(compa)
# compaSum = np.sum(compa, axis=1)
# print(compaSum)
# print(np.outer(compaSum, np.ones(3)))
# print(P0[np.arange(3), compaSum])

# CPGrad = compCPJac(P, tau)
# capProbs = computeCapProbs(P, tau)
# print(capProbs)
# vecCapProbs = np.transpose(capProbs.flatten('F'))
# print(vecCapProbs.shape)
# # minCapProb = np.min(capProbs)
# # mcpLocs = np.argwhere(capProbs == minCapProb)
# # print(capProbs)
# # print(mcpLocs)
# print(vecCapProbs)
# capProbsTest = vecCapProbs.reshape((3, 3), order='F') 
# print(capProbsTest)
# # mcpVecLocs = np.argwhere(vecCapProbs == minCapProb)
# # print(mcpVecLocs)
# # file = open("CPGrad2.txt", "w")
# # file.write("CPGrad = \n")
# # file.write(np.array2string(CPGrad))
# # file.close()

# CODE REMOVED FROM GRAD STEP FUNCTION, NEED TO RECONSIDER: -------------------------------------------------------------------------
# #  after computation of uGradSteps:
# # # project each onto the subspace of zero-sum n^2-dimensional vectors 
#     print("Grads = ")
#     print(uGradSteps*(1/eps))
#     # print("uGradSteps = ")
#     # print(uGradSteps)
#     zsGradSteps = uGradSteps - (1/n**2)*np.matmul(np.ones([n**2, n**2]), uGradSteps)  # [FIX THIS, IT CURRENTLY MAKES THE ENTIRE PVEC ZERO SUM]
#     # print("zsGradSteps = ")
#     # print(zsGradSteps)
#     # compare the min cap probs resulting from each of the above grad steps
#     oldPvec = P.flatten('F')
#     mcpOpts = np.full_like(zsGradSteps, np.nan)
#     # take each of the constrained gradient steps under consideration:
#     newPvecs = np.outer(oldPvec, np.ones([1, mcpLocs.shape[0]])) + zsGradSteps 