# Computation of quantities relevant to optimization of stochastic surveillance strategies
import os
import numpy as np
import math
import time, timeit
from jax import random
import jax
import functools
from jax import grad, jacfwd, jacrev, jit, devices
import jax.numpy as jnp
from StratViz import *

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
    key = jax.random.PRNGKey(1)
    P0 = jnp.zeros_like(A, dtype='float32')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P0 = P0.at[i, j].set(jax.random.uniform(key))
    P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0
    
def initRandPkey(A, key):
    """
    Generate a random initial transition probability matrix with seed s.

    The robot's initial transition probability matrix must be row-stochastic 
    and consistent with the environment graph (described by A) to be valid. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.
    s : int 
        Seed for random number generation.

    Returns
    -------
    numpy.ndarray
        Valid, random initial transition probability matrix. 
    """
    A_shape = jnp.shape(A)
    P0 = jax.random.uniform(key, A_shape)
    P0 = A*P0
    P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0

def initRandPtest(A, key):
    A_shape = jnp.shape(A)
    start_time = time.time()
    P0 = jax.random.uniform(key, A_shape)
    P0 = A*P0
    print("P0 initial creation took %s seconds" % str(time.time() - start_time))
    # print(P0)
    start_time = time.time()
    P0 = projOntoSimplexJIT(P0)
    # P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    # print("P0 row normalization took %s seconds" % str(time.time() - start_time))
    print("P0 simplex projection took %s seconds" % str(time.time() - start_time))
    return P0

def initRandProwtest(A, key):
    A_shape = jnp.shape(A)
    start_time = time.time()
    P0 = jax.random.uniform(key, A_shape)
    P0 = A*P0
    print("P0 initial creation took %s seconds" % str(time.time() - start_time))
    # print(P0)
    start_time = time.time()
    for i in range(A_shape[0]):
        P0 = P0.at[i, :].set(projRowOntoSimplexJIT(P0[i, :]))
    # P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    # print("P0 row normalization took %s seconds" % str(time.time() - start_time))
    print("P0 simplex projection took %s seconds" % str(time.time() - start_time))
    return P0

def initRandPs(A, num):
    """
    Generate a set of `num` random initial transition probability matrices.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.
    num : int 
        Number of initial transition probability matrices to generate.

    Returns
    -------
    numpy.ndarray
        Set of `num` unique, valid, random initial transition probability matrices. 
    
    See Also
    --------
    initRandPseed
    """
    key = jax.random.PRNGKey(0)
    initPs = jnp.zeros((A.shape[0], A.shape[1], num),  dtype='float32')
    for k in range(num):
        key, subkey = jax.random.split(key)
        initPs = initPs.at[:, : , k].set(initRandPkey(A, subkey))
    return initPs

def initRandPsRowTest(A, num):
    key = jax.random.PRNGKey(0)
    initPs_row = jnp.zeros((A.shape[0], A.shape[1], num),  dtype='float32')
    initPs_orig = jnp.zeros((A.shape[0], A.shape[1], num),  dtype='float32')
    for k in range(num):
        key, subkey = jax.random.split(key)
        initPs_row = initPs_row.at[:, : , k].set(initRandProwtest(A, subkey))
        initPs_orig = initPs_orig.at[:, : , k].set(initRandPtest(A, subkey))
    return initPs_row, initPs_orig



def genGraphCode(A):
    bin_string = ""
    for i in range(A.shape[0] - 1):
        for j in range(i + 1, A.shape[1]):
            bin_string = bin_string + str(int(A[i, j]))
    graphNum = int(bin_string, base=2)
    graphCode = str(A.shape[0]) + "_" + str(graphNum)
    return graphCode



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
    A = jnp.identity(n)
    A = A.at[0, :].set(jnp.ones(n))
    A = A.at[:, 0].set(jnp.ones(n))
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
    A = jnp.identity(n)
    A = A + jnp.diag(jnp.ones(n - 1), 1)
    A = A + jnp.diag(jnp.ones(n - 1), -1)
    return A

def genSplitStarG(leftLeaves, rightLeaves, numLineNodes):
    """
    Generate binary adjacency matrix for a "split star" graph. 

    The "split star" graph has a line graph with `numLineNodes` nodes, 
    with one end of the line being connected to an additional `leftLeaves` 
    leaf nodes, and the other end having `rightLeaves` leaf nodes. 

    Parameters
    ----------
    leftLeaves : int 
        Number of leaf nodes on the left end of the line graph.
    rightLeaves : int
        Number of leaf nodes on the right end of the line graph.
    numLineNodes : int
        Number of nodes in the connecting line graph (excluding leaves).
    
    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix for the split star graph. 
    """
    leftStar = jnp.identity(leftLeaves + 1)
    leftStar = leftStar.at[leftLeaves, :].set(jnp.ones(leftLeaves + 1))
    leftStar = leftStar.at[:, leftLeaves].set(jnp.ones(leftLeaves + 1))
    rightStar = genStarG(rightLeaves + 1)
    midLine = genLineG(numLineNodes)

    n = leftLeaves + rightLeaves + numLineNodes
    splitStar = jnp.identity(n)
    splitStar = splitStar.at[0:(leftLeaves + 1), 0:(leftLeaves + 1)].set(leftStar)
    splitStar = splitStar.at[leftLeaves:(leftLeaves + numLineNodes), leftLeaves:(leftLeaves + numLineNodes)].set(midLine)
    splitStar = splitStar.at[(leftLeaves + numLineNodes - 1):n, (leftLeaves + numLineNodes - 1):n].set(rightStar)
    return splitStar
    
def genGridG(width, height):
    n = width*height
    A = jnp.identity(n)
    A = A + jnp.diag(jnp.ones(n - height), height)
    A = A + jnp.diag(jnp.ones(n - height), -height)
    for k in range(n):
        if k % height == 0:
            A = A.at[k, k + 1].set(1)
        elif k % height == (height - 1):
            A = A.at[k, k - 1].set(1)
        else:
            A = A.at[k, k + 1].set(1)
            A = A.at[k, k - 1].set(1)
    return A

def genCycleG(n):
    A = genLineG(n)
    A = A.at[0, n - 1].set(1)
    A = A.at[n - 1, 0].set(1)
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
    n = P.shape[0]
    F = jnp.full((n, n, tau), np.NaN)
    F = F.at[:, :, 0].set(P)
    for i in range(1, tau):
        F = F.at[:, :, i].set(jnp.matmul(P, (F[:, :, i - 1] - jnp.diag(jnp.diag(F[:, :, i - 1])))))
    return F

@functools.partial(jit, static_argnames=['tau'])
def computeFHTProbMatsJIT(P, F0, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Array of `tau` distinct First Hitting Time Probability matrices. 
    """
    F0 = F0.at[:, :, 0].set(P)
    for i in range(1, tau):
        F0 = F0.at[:, :, i].set(jnp.matmul(P, (F0[:, :, i - 1] - jnp.diag(jnp.diag(F0[:, :, i - 1])))))
    return F0

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
    capProbs = jnp.sum(F, axis=2)
    return capProbs

@functools.partial(jit, static_argnames=['tau'])
def computeCapProbsJIT(P, F0, tau):
    """
    Compute Capture Probability Matrix.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix.
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Capture Probability matrix. 
    
    See Also
    --------
    computeFHTProbMatsJIT
    """
    F = computeFHTProbMatsJIT(P, F0, tau)
    capProbs = jnp.sum(F, axis=2)
    return capProbs

@functools.partial(jit, static_argnames=['tau'])
def computeMCPJIT(P, F0, tau):
    """
    Compute Minimum Capture Probability.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Minimum Capture Probability. 
    
    See Also
    --------
    computeCapProbsJIT
    """
    F = computeCapProbsJIT(P, F0, tau)
    mcp = jnp.min(F)
    return mcp

@functools.partial(jit, static_argnames=['tau', 'lcpNum'])
def computeLCPsJIT(P, F0, tau, lcpNum):
    """
    Compute Lowest 'lcpNum' Capture Probabilities.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    lcpNum : int
        Number of the lowest capture probabilities to compute. 
    
    Returns
    -------
    numpy.ndarray
        Set of `lcpNum` lowest capture probabilities. 
    
    See Also
    --------
    computeCapProbsJIT
    """
    F = computeCapProbsJIT(P, F0, tau)
    n = F.shape[0]
    Fvec = F.flatten('F')
    lcps = jnp.sort(Fvec)[0:lcpNum]
    return lcps


def lcpInds(P, F0, tau, lcpNum):
    F = computeCapProbsJIT(P, F0, tau)
    n = F.shape[0]
    Fvec = F.flatten('F')
    lcpInds = jnp.argsort(Fvec)[0:lcpNum]
    return lcpInds

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
    J = jnp.full([n**2, n**2, tau], np.NaN) # initialize array to store Jacobians of FHT probability matrices
    J = J.at[:, :, 0].set(jnp.identity(n**2)) # Jacobian of F_1 is the identity matrix

    # generate matrices needed for intermediary steps in computation of Jacobians:
    Pbar = jnp.zeros([n**2, n**2])
    for i in range(n):
        Pbar = Pbar.at[i*n:(i + 1)*n, i*(n + 1)].set(P[:, i])
    B = jnp.kron(jnp.identity(n), P) - Pbar

    # recursive computation of Jacobians of FHT probability matrices:
    for i in range(1, tau):
        J = J.at[:, :, i].set(jnp.kron(jnp.transpose(F[:, :, i - 1]), jnp.identity(n)) - jnp.kron(jnp.diag(jnp.diag(F[:, :, i - 1])), jnp.identity(n)) + jnp.matmul(B, J[:, :, i - 1]))

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
    CPGrad = jnp.sum(J, axis=2)
    return CPGrad

# Autodiff version of Cap Prob Jacobian computation:
compCPJacAuto = jacfwd(computeCapProbs)
# wrapper function:
def compCPJacJax(P, tau):
    J = compCPJacAuto(P, tau)
    n = P.shape[0]
    J = J.reshape((n**2, n**2), order='F')
    return J

# Autodiff version of Min Cap Prob Gradient computation:
compMCPGradAuto = jacrev(computeMCPJIT)
# wrapper function:
@functools.partial(jit, static_argnames=['tau'])
def compMCPGradJIT(P, F0, tau):
    G = compMCPGradAuto(P, F0, tau)
    n = P.shape[0]
    G = G.reshape((n**2), order='F')
    return G

# Autodiff version of Lowest Cap Probs Gradient computation:
compLCPGradsAuto = jacrev(computeLCPsJIT)
# wrapper function:
@functools.partial(jit, static_argnames=['tau', 'lcpNum'])
def compLCPGradsJIT(P, F0, tau, lcpNum):
    G = compLCPGradsAuto(P, F0, tau, lcpNum)
    n = P.shape[0]
    # G = jnp.reshape(G, (lcpNum, n**2), order='F')
    G = G.reshape((lcpNum, n**2), order='F')
    return G

# Autodiff version of average Lowest Cap Probs Gradient computation:
@functools.partial(jit, static_argnames=['tau', 'lcpNum'])
def compAvgLCPGradJIT(P, F0, tau, lcpNum):
    G = compLCPGradsAuto(P, F0, tau, lcpNum)
    n = P.shape[0]
    G = G.reshape((lcpNum, n**2), order='F')
    avgG = jnp.mean(G, axis=0)
    return avgG

# Autodiff version for JIT compiling:
compCPJacAutoJIT = jacfwd(computeCapProbsJIT)
# wrapper function:
# @jit
@functools.partial(jit, static_argnames=['tau'])
def compCPJacJIT(P, F0, tau):
    J = compCPJacAutoJIT(P, F0, tau)
    n = P.shape[0]
    J = J.reshape((n**2, n**2), order='F')
    return J
    

def getZeroCols(A):
    """
    Identify columns of Jacobian of the Capture Probability Matrix to zero-out.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix corresponding to the environment graph.
    
    Returns
    -------
    numpy.ndarray
        Array containing indices of the columns of the Cap Prob Jacobian to zero-out. 

    See Also
    --------
    zeroCPJacColsJIT
    """
    A = A.flatten('F')
    cols = np.where(A == 0)[0]
    return cols

# @functools.partial(jit, static_argnames=['cols'])
@jit
def zeroJacColsJIT(J, cols):
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
    cols : numpy.ndarray
        Array containing indices of the columns of the Jacobian to set to zero. 
    
    Returns
    -------
    numpy.ndarray
        Jacobian of Capture Probability Matrix with appropriately-zeroed columns. 

    See Also
    --------
    getZeroCols
    """
    J = J.at[:, cols].set(jnp.zeros((J.shape[0], cols.shape[0])))
    return J

# @functools.partial(jit, static_argnames=['cols'])
@jit
def zeroGradColsJIT(mcpGrad, cols):
    """
    Zero-out appropriate columns of gradient of the min capture probability.

    To ensure gradient-based updates to the Transition Probability Matrix do not
    introduce non-zero transition probabilities corresponding to edges that are 
    absent from the environment graph, the corresponding columns of the Jacobian
    of the Capture Probability Matrix are set to zero. For more information, see
    [LINK TO DOCUMENT ON GITHUB].

    Parameters
    ----------
    J : numpy.ndarray 
        Jacobian of the Capture Probability Matrix. 
    cols : numpy.ndarray
        Array containing indices of the columns of the Jacobian to set to zero. 
    
    Returns
    -------
    numpy.ndarray
        Jacobian of Capture Probability Matrix with appropriately-zeroed columns. 

    See Also
    --------
    getZeroCols
    """
    mcpGrad = mcpGrad.at[cols].set(jnp.zeros(cols.shape[0]))
    return mcpGrad

@jit
def projOntoSimplexJIT(P):
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
    sortMapping = jnp.fliplr(jnp.argsort(P, axis=1))
    X = jnp.full_like(P, np.nan)
    for i  in range (n):
        for j in range (n):
            X = X.at[i, j].set(P[i, sortMapping[i, j]])
    Xtmp = jnp.matmul(jnp.cumsum(X, axis=1) - 1, jnp.diag(1/jnp.arange(1, n + 1)))
    rhoVals = jnp.sum(X > Xtmp, axis=1) - 1
    lambdaVals = -Xtmp[jnp.arange(n), rhoVals]
    newX = jnp.maximum(X + jnp.outer(lambdaVals, jnp.ones(n)), jnp.zeros([n, n]))
    newP = jnp.full_like(P, np.nan)
    for i in range(n):
        for j in range(n):
            newP = newP.at[i, sortMapping[i, j]].set(newX[i, j])
    return newP

@jit
def projRowOntoSimplexJIT(row):
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
    n = len(row)
    sortMapping = jnp.flip(jnp.argsort(row))
    X = jnp.full_like(row, np.nan)
    for j in range (n):
        X = X.at[j].set(row[sortMapping[j]])
    Xtmp = jnp.matmul(jnp.cumsum(X) - 1, jnp.diag(1/jnp.arange(1, n + 1)))
    rhoVal = jnp.sum(X > Xtmp) - 1
    lambdaVal = -Xtmp[rhoVal]
    newX = jnp.maximum(X + lambdaVal, jnp.zeros((n)))
    newRow = jnp.full_like(row, np.nan)
    for j in range(n):
        newRow = newRow.at[sortMapping[j]].set(newX[j])
    return newRow

# TESTING -----------------------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)

    # print("Devices available:")
    # print(jax.devices())

    # tau = 2 # attack duration
    # A = genStarG(4)
    # # P0 = initRandPseed(A, 1)
    # n = A.shape[0]
    # F0 = jnp.full((n, n, tau), np.NaN)

    # A = genGridG(3, 4)
    # numInitPs = 3
    # print("REGULAR VS ROW PROJECTION:")
    # initPs_row, initPs_orig= initRandPsRowTest(A, numInitPs)

    # check = jnp.sum(initPs_row - initPs_orig)
    # print("check = ")
    # print(check)
    # print("initPs_row = ")
    # print(initPs_row[:, :, 0])
    # print("initPs_orig = ")
    # print(initPs_orig[:, :, 0])

    graphNum = 1
    numInitPs = 10
    for i in range(1):
        for j in range(1):
            for k in range(20, 21):
                print("-------- Working on Graph Number " + str(graphNum) + "----------")
                # A = genSplitStarG(leftLeavesR[k], rightLeavesR[j], midLineLenR[i])
                # A = genStarG(6)
                start_time = time.time()
                A = genGridG(k, k+1)
                print("Generating bin adj mat took %s seconds" % str(time.time() - start_time))
                n = A.shape[0]
                tau1 = 2*(k - 1) + 1
                F0 = jnp.full((n, n, tau1), np.NaN)
                start_time = time.time()
                initPs = initRandPs(A, numInitPs)
                print("Generating initial P mats took %s seconds" % str(time.time() - start_time))
                for pNum in range(numInitPs):
                    print("----------------------------------------------------")
                    start_time = time.time()
                    grad1 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 1 took %s seconds" % str(time.time() - start_time))
                    start_time = time.time()
                    grad2 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 2 took %s seconds" % str(time.time() - start_time))
                    start_time = time.time()
                    grad3 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 3 took %s seconds" % str(time.time() - start_time))
                    start_time = time.time()
                    grad4 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 4 took %s seconds" % str(time.time() - start_time))
                    print("----------------------------------------------------")
                tau2 = 2*(k - 1) + 3
                print("Working with new tau value ---------------------------")
                for pNum in range(numInitPs):
                    print("----------------------------------------------------")
                    start_time = time.time()
                    grad1 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 1 took %s seconds" % str(time.time() - start_time))
                    start_time = time.time()
                    grad2 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 2 took %s seconds" % str(time.time() - start_time))
                    start_time = time.time()
                    grad3 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 3 took %s seconds" % str(time.time() - start_time))
                    start_time = time.time()
                    grad4 = compMCPGradJIT(initPs[:, :, pNum], F0, tau1)
                    print("grad computation 4 took %s seconds" % str(time.time() - start_time))
                    print("----------------------------------------------------")
                




    # cwd = os.getcwd()
    # # drawEnvGraph(A, 1, cwd)
    # n = A.shape[0]
    # P0 = initRandPseed(A, 2)
    # drawTransProbGraph(A, P0, "graph1_tau5", 0, cwd)

    # print("P0 = ")
    # print(P0)
    # F0 = jnp.full((n, n, tau), np.NaN)
    # lcpNum = 4

    # F = computeCapProbsJIT(P0, F0, tau)
    # Fvec = F.flatten('F')
    # print("Fvec = ")
    # print(Fvec)
    # # FHTs = computeFHTProbMatsJIT(P0, F0, tau)
    # # printFHTProbMats(FHTs)
    # lcps = computeLCPsJIT(P0, F0, tau, lcpNum)
    # lcpIndices = lcpInds(P0, F0, tau, lcpNum)
    # print("lcps = ")
    # print(lcps)
    # print("lcpIndices = ")
    # print(lcpIndices)

    # lcpGrads = compLCPGradsJIT(P0, F0, tau, lcpNum)
    # avgLCPGrad = compAvgLCPGradJIT(P0, F0, tau, lcpNum)
    # print("avgLCPGrad = ")
    # print(avgLCPGrad)
    # J = compCPJacJIT(P0, F0, tau)
    # avgFromJ = jnp.zeros(n**2)
    # for k in range(lcpNum):
    #     avgFromJ = avgFromJ + J[lcpIndices[k], :]
    # avgFromJ = avgFromJ/lcpNum
    # print("avgFromJ = ")
    # print(avgFromJ)
    # meanDiff = jnp.mean(avgFromJ - avgLCPGrad)
    # print("meanDiff = ")
    # print(meanDiff)

    
 
    # zeroRows, zeroCols = np.where(np.equal(F, np.zeros((n, n))))
    # print("zeroInds = ")
    # for i in range(len(zeroRows)):
    #     print(str(zeroRows[i]) + ", " + str(zeroCols[i]))

    # # FHT = computeFHTProbMatsJIT(P0, F0, tau)
    # start_time = time.time()
    # F1 = computeMCPJIT(P0, F0, tau).block_until_ready()
    # print("MCP computation round 1 took %s seconds ---" %(time.time() - start_time))
    # start_time = time.time()
    # F2 = computeMCPJIT(P0, F0, tau).block_until_ready()
    # print("MCP computation round 2 took %s seconds ---" %(time.time() - start_time))
    # start_time = time.time()
    # J1 = compMCPGradJIT(P0, F0, tau).block_until_ready()
    # print("MCP grad computation round 1 took %s seconds ---" %(time.time() - start_time))
    # start_time = time.time()
    # J2 = compMCPGradJIT(P0, F0, tau).block_until_ready()
    # print("MCP grad computation round 2 took %s seconds ---" %(time.time() - start_time))



    # print("A = ")
    # print(A)
    # print("vectorized A = ")
    # print(A.flatten('F'))

    # start_time = time.time()
    # cols = getZeroCols(A)
    # print("Column finding with np.where took %s seconds ---" %(time.time() - start_time))
    # # print("cols = ")
    # # print(cols)
    

    # # print("cols2 = ")
    # # print(cols2)

    # # print("P0 = ")
    # # print(P0)

    # # J = compCPJac(P0, tau)
    # # print("J by hand = ")
    # # print(J)

    # J = compCPJacJIT(P0, F0)
    # # # print("J by autodiff = ")
    # # # print(J)
    # # # print(J.shape)

    # start_time = time.time()
    # J1 = zeroJacColsJIT(J, cols)
    # print("Column zeroing took %s seconds ---" %(time.time() - start_time))

    # start_time = time.time()
    # J2 = zeroJacColsJIT(J, cols)
    # print("Column zeroing round 2 took %s seconds ---" %(time.time() - start_time))
    # # print("J with zeroed columns = ")
    # # print(J1)


    # # J = compCPJacJIT(P0, tau, F0)
    # J = compCPJacJIT(P0, F0)
    # print("J by autodiff with JIT = ")
    # print(J)

    # n = P0.shape[0]
    # F = computeCapProbs(P0, tau)
    # F = F.reshape((n**2,), order='F')
    # print("F = ")
    # print(F)
    # mcploc = jnp.argwhere(F == jnp.min(F))
    # print("mcploc = ")
    # print(mcploc)

    # print("Jacobian row at MCP = ")
    # print(J[mcploc, :])

    # mcp = computeMCPJIT(P0, F0)
    # print("MCP = ")
    # print(mcp)

    # G = compMCPGradAuto(P0, F0)
   
    # G = G.reshape((n**2), order='F')
    # print("MCP Grad = ")
    # print(G)
    # print(G.shape)
    # Jrow = J[mcploc, :].reshape(n**2)
    # print(Jrow.shape)


    # start_time = time.time()
    # J = compCPJacJIT(P0, F0)
    # mcploc = jnp.argwhere(F == jnp.min(F))
    # Jrow = J[mcploc, :]
    # print("--- Automatic Forward-Mode MCP Gradient Computation took: %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # G = compMCPGradAuto(P0, F0)
    # print("--- Automatic Reverse-Mode MCP Gradient Computation took: %s seconds ---" % (time.time() - start_time))
    
    
    # print("mcploc = ")
    # print(mcploc)
    # Jrow = Jrow.reshape(n**2)
    # G = G.reshape((n**2), order='F')
    # print("MCP grads l2-norm error = ")
    # e = Jrow - G
    # print(jnp.dot(e, e))

    # P1 = initRandPseed(A, 2)

    # start_time = time.time()
    # J = compCPJacJIT(P0, F0)
    # mcploc = jnp.argwhere(F == jnp.min(F))
    # Jrow = J[mcploc, :]
    # print("--- Automatic Forward-Mode MCP Gradient Computation round 2 took: %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # G = compMCPGradAuto(P0, F0)
    # print("--- Automatic Reverse-Mode MCP Gradient Computation round 2 took: %s seconds ---" % (time.time() - start_time))


    # print("mcploc = ")
    # print(mcploc)
    # Jrow = Jrow.reshape(n**2)
    # G = G.reshape((n**2), order='F')
    # print("MCP grads l2-norm error = ")
    # e = Jrow - G
    # print(jnp.dot(e, e))

    # start_time = time.time()
    # J = compCPJac(P0, tau)
    # print("--- By-hand Jacobian Computation round 2 took: %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()

    # start_time = time.time()
    # J = compCPJacJax(P0, tau)
    # print("--- Automatic Jacobian Computation took: %s seconds ---" % (time.time() - start_time))
    # start_time = time.time()


    # start = timeit.default_timer()
    # J = compCPJacJax(P0, tau).block_until_ready()
    # print('Automatic Jacobian Computation round 2 took: {} seconds'.format(timeit.default_timer() - start))


    # start = timeit.default_timer()
    # P0_gpu = jax.device_put(P0)
    # F0_gpu = jax.device_put(F0)
    # tau_gpu = jax.device_put(tau)
    # print('GPU copying Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # start = timeit.default_timer()
    # F1 = computeMCPJIT(P0, F0, tau).block_until_ready()
    # print('MCP computation warmup Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # start = timeit.default_timer()
    # F1 = computeMCPJIT(P0, F0, tau).block_until_ready()
    # print('MCP computation round 1 Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # start = timeit.default_timer()
    # J = compMCPGradJIT(P0_gpu, F0_gpu, tau).block_until_ready()
    # print('Rev Mode MCP Grad JIT warmup Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # start = timeit.default_timer()
    # J1 = compMCPGradJIT(P0_gpu, F0_gpu, tau).block_until_ready()
    # print('Rev Mode MCP Grad JIT round 1 Elapsed time: {} seconds'.format(timeit.default_timer() - start))


    # deltaP = 0.2*initRandP(np.ones([4, 4]))
    # Ptest = P + deltaP
    # print("Ptest = ")
    # print(Ptest)
    # newP = projOntoSimplex(Ptest)
    # print("newP = ")
    # print(newP)



