# Computation of quantities relevant to optimization of stochastic surveillance strategies
import os
import numpy as np
import math
import time, timeit
import jax
from jax import random
import functools
from jax import grad, jacfwd, jacrev, jit, devices, lax
import jax.numpy as jnp
from StratViz import *

def initRandP(A):
    """
    Generate a random initial transition probability matrix.

    The robot's initial transition probability matrix must be row-stochastic 
    and consistent with the environment graph (described by `A`) to be valid. 
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
    key = random.PRNGKey(1)
    P0 = jnp.zeros_like(A, dtype='float32')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P0 = P0.at[i, j].set(random.uniform(key))
    P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0
    
def initRandPkey(A, key):
    """
    Generate a random initial transition probability matrix using PRNG key `key`.

    The robot's initial transition probability matrix must be row-stochastic 
    and consistent with the environment graph (described by `A`) to be valid. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.
    key : int 
        Jax PRNGKeyArray for random number generation.

    Returns
    -------
    numpy.ndarray
        Valid, random initial transition probability matrix. 
    """
    A_shape = jnp.shape(A)
    P0 = random.uniform(key, A_shape)
    P0 = A*P0
    P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
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
    key = random.PRNGKey(0)
    initPs = jnp.zeros((A.shape[0], A.shape[1], num),  dtype='float32')
    for k in range(num):
        key, subkey = random.split(key)
        initPs = initPs.at[:, : , k].set(initRandPkey(A, subkey))
    return initPs


def genGraphCode(A):
    """
    Generate a unique code representing the environment graph.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.

    Returns
    -------
    String
        Unique encoding of the binary adjacency matrix.
    
    """
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
    """
    Generate binary adjacency matrix for a grid graph. 

    Parameters
    ----------
    width : int 
        Number of nodes in each row of the grid graph.
    height : int
        Number of nodes in each column of the grid graph.
    
    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix for the grid graph. 
    """
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
    """
    Generate binary adjacency matrix for a cycle graph. 

    Parameters
    ----------
    n: int 
        Number of nodes in the cycle graph.

    Returns
    -------
    numpy.ndarray
        Binary adjacency matrix for the cycle graph. 
    """
    A = genLineG(n)
    A = A.at[0, n - 1].set(1)
    A = A.at[n - 1, 0].set(1)
    return A


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
    Compute Lowest `lcpNum` Capture Probabilities.

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
    Fvec = F.flatten('F')
    lcps = jnp.sort(Fvec)[0:lcpNum]
    return lcps


# Autodiff version of Min Cap Prob Gradient computation:
compMCPGradAuto = jacrev(computeMCPJIT)
# wrapper function:
@functools.partial(jit, static_argnames=['tau'])
def compMCPGradJIT(P, A, F0, tau):
    G = compMCPGradAuto(P, F0, tau)
    G = G*A
    return G

# Autodiff version of Lowest Cap Probs Gradient computation:
compLCPGradsAuto = jacrev(computeLCPsJIT)
# wrapper function:
@functools.partial(jit, static_argnames=['tau', 'lcpNum'])
def compLCPGradsJIT(P, A, F0, tau, lcpNum):
    G = compLCPGradsAuto(P, F0, tau, lcpNum)
    G = G*A
    return G

# Autodiff version of average Lowest Cap Probs Gradient computation:
@functools.partial(jit, static_argnames=['tau', 'lcpNum'])
def compAvgLCPGradJIT(P, A, F0, tau, lcpNum):
    G = compLCPGradsAuto(P, F0, tau, lcpNum)
    G = G*A
    avgG = jnp.mean(G, axis=0)
    return avgG

# Parametrization of the P matrix
@functools.partial(jit, static_argnames=['tau'])
def getPfromParam(Q, A):
    P = Q*A
    P = jnp.maximum(jnp.zeros_like(P), P) # apply component-wise ReLU
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize to generate valid prob dist
    return P

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau'])
def loss(Q, A, F0, tau):
    P = getPfromParam(Q, A)
    mcp = computeMCPJIT(P, F0, tau)
    return mcp

# Autodiff parametrized loss function
compLossGradAuto = jacrev(loss)
@functools.partial(jit, static_argnames=['tau'])
def compLossGrad(Q, A, F0, tau):
    G = compLossGradAuto(Q, A, F0, tau) 
    return G
    

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


def projRowOntoSimplexTest(row):
    """
    Project rows of the Transition Probability Matrix `P` onto the probability simplex.

    To ensure gradient-based updates to the Transition Probability Matrix maintain
    row-stochasticity, the rows of the updated Transition Probability Matrix are projected 
    onto the nearest point on the probability n-simplex, where `n` is the number of
    columns of `P`.  For further explanation, see [LINK TO DOCUMENT ON GITHUB], and 
    for more about the projection algorithm used, see http://www.optimization-online.org/DB_FILE/2014/08/4498.pdf.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition Probability Matrix after gradient update, potentially invalid. 
    
    Returns
    -------
    numpy.ndarray
        Valid Transition Probability Matrix nearest to `P` in Euclidian sense. 
    """
    v = []
    vt = []
    v.append(row[0])
    rho = row[0] - 1
    # FIRST PASS:
    for k in range(1, len(row)):
        yn = row[k]
        if yn > rho:
            rho = rho + (yn - rho)/(len(v) + 1)
            if rho > (yn - 1):
                v.append(yn)
            else:
                vt.append(v)
                v.clear()
                v.append(yn)
                rho = yn - 1
    # CLEANUP PASS:
    for k in range(len(vt)):
        yn = vt[k]
        if yn > rho:
            v.append(yn)
            rho = rho + (yn - rho)/(len(v))
    # ELEMENT ELIMINATION LOOP:
    lenChange = True
    while lenChange:
        cardV = len(v)
        k = 0
        while k < len(v):
            if v[k] <= rho:
                y = v.pop(k)
                rho = rho + (rho - y)/(len(v))
            else:
                k = k + 1
        if cardV == len(v):
            lenChange = False
    tau = rho
    # PROJECTION:
    row = jnp.maximum(row - tau, jnp.zeros(jnp.shape(row)))
    return row

# @jit
# def projRowOntoSimplexMichelot(row):
#     """
#     Project rows of the Transition Probability Matrix `P` onto the probability simplex.

#     To ensure gradient-based updates to the Transition Probability Matrix maintain
#     row-stochasticity, the rows of the updated Transition Probability Matrix are projected 
#     onto the nearest point on the probability n-simplex, where `n` is the number of
#     columns of `P`.  For further explanation, see [LINK TO DOCUMENT ON GITHUB], and 
#     for more about the projection algorithm used, see http://www.optimization-online.org/DB_FILE/2014/08/4498.pdf.

#     Parameters
#     ----------
#     P : numpy.ndarray 
#         Transition Probability Matrix after gradient update, potentially invalid. 
    
#     Returns
#     -------
#     numpy.ndarray
#         Valid Transition Probability Matrix nearest to `P` in Euclidian sense. 
#     """
#     v = row
#     print("row shape = ")
#     print(jnp.shape(row))
#     print("len v = ")
#     print(len(v))
#     rho = (jnp.sum(row) - 1)/len(row)
#     # ELEMENT ELIMINATION LOOP:
#     lenChange = True
#     while lenChange:
#         cardV = len(v)
#         k = 0
#         # while k < len(v):
#         #     lax.cond(v[k] <= rho, lambda v, k : jnp.delete(v, k), lambda k : k + 1, v, k)
#         #     # ^THINK ABOUT HOW TO DO THIS NOT ELEMENT-WISE
#         #     # if v[k] <= rho:
#         #     #     jnp.delete(v, k)
#         #     # else:
#         #     #     k = k + 1
#         # rho = (jnp.sum(v) - 1)/(len(v))
#         v, rho = elimElts(v, rho)
#         if cardV == len(v):
#             lenChange = False
#     tau = rho
#     # PROJECTION:
#     row = jnp.maximum(row - tau, jnp.zeros(jnp.shape(row)))
#     return row

# @jit
# def elimElts(v, rho):
#     rhoVals = jnp.full_like(v, rho)
    
#     delInds = jnp.argwhere(v <= rhoVals)
#     newV = jnp.delete(v, delInds)
#     newRho = (jnp.sum(newV) - 1)/(len(newV))
#     return newV, newRho


# TESTING -----------------------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)

    # print("Devices available:")
    # print(jax.devices())


    key = random.PRNGKey(1)
    print(type(key))
    print(key)
    newkey, subkey = random.split(key)
    print(type(newkey))
    print(type(subkey))
    print(newkey)
    print(subkey)


    # A = genStarG(6)
    # cols = getZeroCols(A)
    # n = A.shape[0]
    # P0 = initRandP(A)
    # randP = initRandP(A) + jnp.full_like(A, 0.1)
    # tau = 4
    # F0 = jnp.full((n, n, tau), np.NaN)
    # P = P0
    # colCompTimes = []
    # AcompTimes = []
    # lcpNum = 4

    # numInitPs = 2
    # print("ALGORITHM COMPARISON:")

    # A_shape = jnp.shape(A)
    # n = A_shape[0]
    # initPs = jnp.full((n, n, numInitPs), np.NaN)
    # for k in range(numInitPs):
    #     key = random.PRNGKey(k)
    #     P0 = random.uniform(key, A_shape)
    #     initPs = initPs.at[:, :, k].set(P0)

    # for k in range(numInitPs):
    #     print("---------------------------------------------------------------")
    #     P0 = initPs[:, :, k]
    #     # print("P0 = ")
    #     # print(P0)
    #     projPOrig1 = jnp.zeros(A_shape)
    #     start_time = time.time()
    #     for i in range(jnp.shape(P0)[0]):
    #         row = P0[i, :]
    #         projPOrig1 = projPOrig1.at[i, :].set(projRowOntoSimplexJIT(row))
    #     print("Original row-wise jitted projection algorithm took %s seconds" % str(time.time() - start_time))
    #     # print("projPOrig1 = ")
    #     # print(projPOrig1)

    #     start_time = time.time()
    #     projPOrig2 = projOntoSimplexJIT(P0)
    #     print("Original full-matrix jitted projection algorithm took %s seconds" % str(time.time() - start_time))
    #     # print("projPOrig2 = ")
    #     # print(projPOrig2)

    #     projPTest = jnp.zeros(A_shape)
    #     start_time = time.time()
    #     for i in range(jnp.shape(P0)[0]):
    #         row = P0[i, :]
    #         projPTest = projPTest.at[i, :].set(projRowOntoSimplexTest(row))
    #     print("New projection algorithm took %s seconds" % str(time.time() - start_time))
    #     # print("projPTest = ")
    #     # print(projPTest)

    #     # projPMichelotTest = jnp.zeros(A_shape)
    #     # start_time = time.time()
    #     # for i in range(jnp.shape(P0)[0]):
    #     #     row = P0[i, :]
    #     #     projPMichelotTest = projPTest.at[i, :].set(projRowOntoSimplexMichelot(row))
    #     # print("Michelot jitted projection algorithm took %s seconds" % str(time.time() - start_time))
    #     # print("projPTest = ")
    #     # print(projPTest)

    #     check1 = jnp.sum(projPOrig1 - projPTest)
    #     print("check1 = " + str(check1))
    #     check2 = jnp.sum(projPOrig2 - projPTest)
    #     print("check2 = " + str(check2))
    #     # check3 = jnp.sum(projPMichelotTest - projPTest)
    #     # print("check3 = " + str(check3))
    #     print("---------------------------------------------------------------")

    
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




