# Optimization of the performance of stochastic surveillance strategies

import numpy as np
np.set_printoptions(linewidth=np.inf)
import math
from StratComp import *

P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # initialize transition prob matrix
# P = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # initialize transition prob matrix
P = np.matmul(np.diag(1/np.sum(P, axis=1)), P)   # normalize to generate valid prob dist

tau = 3  # attack duration

def compFkGrads(P, tau):
    n = P.shape[0]  # get number of nodes
    F = computeFHTProbMats(P, tau) # get first hitting time probability matrices
    J = np.full([n**2, n**2, tau], np.NaN) # initialize array to store gradients of FHT probability matrices
    J[:, :, 0] = np.identity(n**2) # gradient of F_1 is the identity matrix

    # initialize matrices needed for gradient computations:
    PBar = np.zeros([n**2, n**2])
    for i in range(n):
        PBar[i*n:(i + 1)*n, i*(n + 1)] = P[:, i]

    B = np.kron(np.identity(n), P) - PBar

    # file = open("variables.txt", "w")
    # file.write("B = \n")
    # file.write(np.array2string(B) + "\n")
    # file.write("J_0 = \n")
    # file.write(np.array2string(J[:, :, 0]) + "\n")

    # recursive computating of gradients of FHT probability matrices:
    for i in range(1, tau):
        J[:, :, i] = np.kron(np.transpose(F[:, :, i - 1]), np.identity(n)) - np.kron(np.diag(np.diag(F[:, :, i - 1])), np.identity(n)) + B*J[:, :, i - 1]

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

# Sums gradients of FHT probability matrices to get gradient of Cap Prob Matrix
def compCPGrad(P, tau):
    J = compFkGrads(P, tau)
    CPGrad = np.sum(J, axis=2)
    return CPGrad
    

# TESTING ------------------------------------------------------------------------

CPGrad = compCPGrad(P, tau)

# file = open("CPGrad.txt", "w")
# file.write("CPGrad = \n")
# file.write(np.array2string(CPGrad))
# file.close()

