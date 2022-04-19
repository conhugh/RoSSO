# Computation of quantities relevant to optimization of stochastic surveillance strategies
import numpy as np
import math
import random

def initRandP(A):
    """
    Generate a random initial transition probability matrix.

    The robot's initial transition probability matrix must be row-stochastic 
    and consistent with the environment graph (described by A) to be valid. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.
    
    Returns
    -------
    numpy.ndarray
        Valid, random initial transition probability matrix. 
    """
    random.seed(1)
    P0 = np.zeros_like(A, dtype='float64')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P0[i, j] = random.random()
    P0 = np.matmul(np.diag(1/np.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0
    
def initRandPseed(A, s):
    """
    Generate a random initial transition probability matrix with seed s.

    The robot's initial transition probability matrix must be row-stochastic 
    and consistent with the environment graph (described by A) to be valid. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.
    
    Returns
    -------
    numpy.ndarray
        Valid, random initial transition probability matrix. 
    """
    random.seed(s)
    P0 = np.zeros_like(A, dtype='float64')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P0[i, j] = random.random()
    P0 = np.matmul(np.diag(1/np.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0
    

def genStarG(n):
    """
    Generate binary adjacency matrix for a star graph with n nodes.

    Parameters
    ----------
    n : int 
        Number of nodes in the star graph.
    
    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix for the star graph with n nodes. 
    """
    A = np.identity(n)
    A[0, :] = np.ones(n)
    A[:, 0] = np.ones(n)
    return A


def genLineG(n):
    """
    Generate binary adjacency matrix for a line graph with n nodes.

    Parameters
    ----------
    n : int 
        Number of nodes in the line graph.
    
    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix for the line graph with n nodes. 
    """
    A = np.identity(n)
    A = A + np.diag(np.ones(n - 1), 1)
    A = A + np.diag(np.ones(n - 1), -1)
    return A


def computeFHTProbMats(P, tau):
    """
    Compute First Hitting Time Probability matrices.

    To compute the Capture Probability Matrix, we must sum the First 
    Hitting Time Probability matrices from 1 up to `tau` time steps. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Array of `tau` distinct First Hitting Time Probability matrices. 
    """
    F = np.full([P.shape[0], P.shape[1], tau], np.NaN)
    F[:, :, 0] = P
    for i in range(1, tau):
        F[:, :, i] = np.matmul(P, (F[:, :, i - 1] - np.diag(np.diag(F[:, :, i - 1]))))
    return F


def computeCapProbs(P, tau):
    """
    Compute Capture Probability Matrix.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Capture Probability matrix. 
    
    See Also
    --------
    computeFHTProbMats
    """
    F = computeFHTProbMats(P, tau)
    capProbs = np.sum(F, axis=2)
    return capProbs


def printFHTProbMats(F):
    """
    Print the First Hitting Time Probability matrices nicely. 

    Parameters
    ----------
    F : numpy.ndarray 
        Array of First Hitting Time Probability matrices. 
    """
    for i in range(F.shape[2]):
        print("F_" + str(i + 1) + " = " + "\n" + str(F[:, :, i]))



def compFkJacs(P, tau):
    """
    Compute Jacobians of First Hitting Time Probability matrices.

    To compute the Jacobian of the Capture Probability Matrix entries 
    with respect to the Transition Probability Matrix entries, we sum
    the Jacobians of the `tau` distinct First Hitting Time Probability 
    Matrices with respect to the Transition Probability Matrix entries.
    For more information see [LINK TO DOCUMENT ON GITHUB].

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Array of Jacobians of the First Hitting Time Probability matrices. 
    
    See Also
    --------
    compCPJac
    """
    n = P.shape[0]  
    F = computeFHTProbMats(P, tau) 
    J = np.full([n**2, n**2, tau], np.NaN) # initialize array to store Jacobians of FHT probability matrices
    J[:, :, 0] = np.identity(n**2) # Jacobian of F_1 is the identity matrix

    # generate matrices needed for intermediary steps in computation of Jacobians:
    Pbar = np.zeros([n**2, n**2])
    for i in range(n):
        Pbar[i*n:(i + 1)*n, i*(n + 1)] = P[:, i]
    B = np.kron(np.identity(n), P) - Pbar

    # recursive computation of Jacobians of FHT probability matrices:
    for i in range(1, tau):
        J[:, :, i] = np.kron(np.transpose(F[:, :, i - 1]), np.identity(n)) - np.kron(np.diag(np.diag(F[:, :, i - 1])), np.identity(n)) + np.matmul(B, J[:, :, i - 1])

    return J


def compCPJac(P, tau):
    """
    Compute Jacobian of Capture Probability Matrix entries.

    Sum Jacobians of FHT probability matrices to get Jacobian of Capture
    Probability Matrix entries w.r.t. the Transition Probability Matrix entries.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Jacobian of the Capture Probability Matrix entries. 

    See Also
    --------
    compFkJacs
    """
    J = compFkJacs(P, tau)
    CPGrad = np.sum(J, axis=2)
    return CPGrad


def zeroCPJacCols(J, A):
    """
    Zero-out appropriate columns of Jacobian of the Capture Probability Matrix.

    To ensure gradient-based updates to the Transition Probability Matrix do not
    introduce non-zero transition probabilities corresponding to edges that are 
    absent from the environment graph, the corresponding columns of the Jacobian
    of the Capture Probability Matrix are set to zero. For more information, see
    [LINK TO DOCUMENT ON GITHUB].

    Parameters
    ----------
    J : numpy.ndarray 
        Jacobian of the Capture Probability Matrix. 
    A : numpy.ndarray
        Binary adjacency matrix describing the environment graph. 
    
    Returns
    -------
    numpy.ndarray
        Jacobian of Capture Probability Matrix with appropriately-zeroed columns. 

    See Also
    --------
    compCPJac
    """
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i, j] == 0:
                J[:, i*n + j] = np.zeros([n**2])
    return J


def projOntoSimplex(P):
    """
    Project rows of the Transition Probability Matrix `P` onto the probability simplex.

    To ensure gradient-based updates to the Transition Probability Matrix maintain
    row-stochasticity, the rows of the updated Transition Probability Matrix are projected 
    onto the nearest point on the probability n-simplex, where `n` is the number of
    columns of `P`.  For further explanation, see [LINK TO DOCUMENT ON GITHUB], and 
    for more about the projection algorithm used, see https://arxiv.org/abs/1309.1541.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition Probability Matrix after gradient update, potentially invalid. 
    
    Returns
    -------
    numpy.ndarray
        Valid Transition Probability Matrix nearest to `P` in Euclidian sense. 
    """
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
if __name__ == '__main__':
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

    # A = genStarG(6)
    # P = initRandP(A)
    # print("P = ")
    # print(P)
    # deltaP = 0.2*initRandP(np.ones([6, 6]))
    # Ptest = P + deltaP
    # print("Ptest = ")
    # print(Ptest)
    # newP = projOntoSimplex(Ptest)
    # print("newP = ")
    # print(newP)

    A = genStarG(9)
    print(A)
    A = genLineG(9)
    print(A)


