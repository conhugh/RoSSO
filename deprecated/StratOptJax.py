# Optimization of the performance of stochastic surveillance strategies
import numpy as np
import time
from collections import deque
import math
import functools
from StratCompJax import *
from StratViz import *

# Get P update suggested by gradient of min cap prob, apply constraints,
# and return the corresponding updated transition probability matrix P
@functools.partial(jit, static_argnames=['tau'])
def update(P, F0, tau, cols, eps):
    n = P.shape[0]
    grad = compMCPGradJIT(P, F0, tau)
    uGradStep = eps*zeroGradColsJIT(grad, cols)
    newP = P + uGradStep.reshape((n, n), order='F')
    newP = projOntoSimplexJIT(newP)
    return newP


# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
# @functools.partial(jit, static_argnames=['tau', 'eps', 'iters'])
def gradAscentFixed(P0, tau, cols, eps, iters):
    n = P0.shape[0]
    P = P0
    F0 = jnp.full((n, n, tau), 0, dtype='float32')
    for k in range(iters + 1):
        # print("iteration number: " + str(k))
        start_time = time.time()
        P = update(P, F0, tau, cols, eps).block_until_ready()
        print("--- Updating P took: %s seconds ---" % (time.time() - start_time))
        # print("P = ")
        # print(P)
        # input("Press enter to continue...")
        if k % 500 == 0:
            F = computeCapProbsJIT(P, F0, tau).block_until_ready()
            print("Minimum Capture Probability at iteration " + str(k) + ":")
            print(jnp.min(F).block_until_ready())
            # print("P at iteration " + str(k) + ":")
            # print(P)
    F = computeCapProbsJIT(P, F0, tau)
    return P, F


# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscentPConv(P0, tau, cols, eps0, radius):
    print("Running!")
    epsThresh = 0.00005  # threshold (i.e., minimum) step size to be used
    thresh = 5000  # step number at which step size reaches threshold value
    iter = 0  # number of gradient ascent steps taken so far
    n = P0.shape[0]
    F0 = jnp.full((n, n, tau), 0, dtype='float32')
    P = P0 
    avgP = P  # running avg capture probability matrix, for checking convergence
    converged = False
    while not converged:
        # print("iteration number: " + str(iter))
        # set step size
        # start_time = time.time()
        if iter <= thresh:
            eps = (1 - iter/thresh)*eps0 + (iter/thresh)*epsThresh
        else:
            eps = epsThresh
        # print("--- Updating epsilon took: %s seconds ---" % (time.time() - start_time))
        # take gradient ascent step:
        # start_time = time.time()
        P = update(P, F0, tau, cols, eps).block_until_ready()
        # print("--- Updating P took: %s seconds ---" % (time.time() - start_time))
        # print("P = ")
        # print(P)
        # print("F = ")
        # print(F)
        # wait = input("Press enter to continue...")
        # print status info to terminal:
        if (iter % 100) == 0:
            F = computeCapProbsJIT(P, F0, tau).block_until_ready()
            print("Minimum Capture Probability at iteration " + str(iter) + ":")
            print(jnp.min(F))
            # print("P at iteration " + str(iter) + ":")
            # print(P)
            # print("F at iteration " + str(iter) + ":")
            # print(F)
        # check for convergence, update running avg cap probs and step counter:
        # start_time = time.time()
        newAvgP = ((iter)*avgP + P)/(iter + 1)
        diffAvgP = jnp.abs(newAvgP - avgP)
        converged = jnp.amax(diffAvgP) < radius
        avgP = newAvgP
        iter = iter + 1
        # print("--- Convergence check took: %s seconds ---" % (time.time() - start_time))
    F = computeCapProbsJIT(P, F0, tau).block_until_ready()
    print("Minimum Capture Probability at iteration " + str(iter - 1) + ":")
    print(np.min(F))
    # print("P at iteration " + str(iter) + ":")
    # print(P)
    # print("Final diffAvgF = ")
    # print(diffAvgP)
    return P, F
# TESTING ------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)
    A = jnp.array([[1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 1]])

    # A = genStarG(6)
    # A2 = genLineG(10)
    cols = getZeroCols(A)
    # cols = getZeroCols(A2)
    P0_1 = initRandPseed(A, 1)
    P0_2 = initRandPseed(A, 2)
    P0_3 = initRandPseed(A, 3)
    P0_4 = initRandPseed(A, 4)
    # P0 = initRandPseed(A2, 1)
    tau = 8
    eps = 0.01

    iters = 5000
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), 0, dtype='float32')
    
    eps0 = 0.05
    rad = 0.00001
    start_time = time.time()
    [P, F] = gradAscentPConv(P0_1, tau, cols, eps0, rad)
    print("P0 = ")
    print(P0_1)
    print("P_final = ")
    print(P)
    print("--- Optimization took: %s seconds ---" % (time.time() - start_time))


    # start_time = time.time()
    # [P, F] = gradAscentFixed(P0_1, tau, cols, eps, iters)
    # print("P0 = ")
    # print(P0_1)
    # print("Pfinal = ")
    # print(P)    
    # print("--- Optimization took: %s seconds ---" % (time.time() - start_time))



    # [initPMats, optPMats, minCapProbs] = exploreOptima(A, tau, trials)

    # for i in range(trials):
    #     print("P0 = ")
    #     print(initPMats[:, :, i])
    #     print("Pfinal = ")
    #     print(optPMats[:, :, i])
    #     print("Min Cap Prob = ")
    #     print(minCapProbs[i])

    # plotTransProbs2D(initPMats, "Initial P Matrices")

    # plotTransProbs2D(optPMats, "P Matrices at convergence, with P conv radius 0.00001")

    # F = computeCapProbsJIT(P0, F0, tau)
    # print("F = ")
    # print(F)

    # start_time = time.time()
    # [P, F] = gradAscentFixed(P0_2, tau, cols, eps, iters)
    # print("P0 = ")
    # print(P0_2)
    # print("Pfinal = ")
    # print(P)
    
    # print("--- Optimization took: %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # [P, F] = gradAscentFixed(P0_3, tau, cols, eps, iters)
    # print("P0 = ")
    # print(P0_3)
    # print("Pfinal = ")
    # print(P)
    
    # print("--- Optimization took: %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # [P, F] = gradAscentFixed(P0_4, tau, cols, eps, iters)
    # print("P0 = ")
    # print(P0_4)
    # print("Pfinal = ")
    # print(P)
    
    # print("--- Optimization took: %s seconds ---" % (time.time() - start_time))

    # start = timeit.default_timer()
    # update(P0, F0, tau, cols, eps).block_until_ready()
    # print('Update before GPU copying, warmup Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # start = timeit.default_timer()
    # update(P0, F0, tau, cols, eps).block_until_ready()
    # print('Update before GPU copying, round 1 Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # print("P0 type = ")
    # print(type(P0))
    # print("F0 type = ")
    # print(type(F0))
    # print("tau type = ")
    # print(type(tau))
    # print("cols type = ")
    # print(type(cols))

    # start = timeit.default_timer()
    # P0_gpu = jax.device_put(P0)
    # F0_gpu = jax.device_put(F0)
    # tau_gpu = jax.device_put(tau)
    # cols_gpu = jax.device_put(cols)
    # eps_gpu = jax.device_put(eps)
    # iters_gpu = jax.device_put(iters)
    # print('GPU copying Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # print("P0_gpu type = ")
    # print(type(P0_gpu))
    # print("tau_gpu type = ")
    # print(type(tau_gpu))
    # print("cols_gpu type = ")
    # print(type(cols_gpu))

    # start = timeit.default_timer()
    # update(P0_gpu, F0_gpu, tau, cols_gpu, eps).block_until_ready()
    # print('Update after GPU copying, round 1 Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # start = timeit.default_timer()
    # update(P0_gpu, F0_gpu, tau, cols_gpu, eps).block_until_ready()
    # print('Update after GPU copying, round 2 Elapsed time: {} seconds'.format(timeit.default_timer() - start))

    # start = timeit.default_timer()
    # P, F = gradAscentFixed(P0_gpu, tau_gpu, cols_gpu, eps_gpu, iters_gpu).block_until_ready()
    # print('Optimization with JIT Elapsed time: {} seconds'.format(timeit.default_timer() - start))


    # # [P, F] = gradAscentMCPConv(P0, A, tau, 0.05, 0.0000005)
    # # [P, F] = gradAscentFConv(P0, A, tau, 0.05, 0.00001)
    # start_time = time.time()
    # [P, F] = gradAscentPConv(P0, A, tau, 0.05, 0.00001)

    # print("--- Optimization took: %s seconds ---" % (time.time() - start_time))
    # print("P0 = ")
    # print(P0)
    # print("P_final = ")
    # print(P)

    # # print("F_initial = ")
    # F0 = computeCapProbs(P0, tau)
    # # print(F0)
    # # print("F_final = ")
    # # print(F)

    # Pdiff = P - P0
    # Fdiff = F - F0
    # print("P_diff = ")
    # print(Pdiff)
    # print("F_diff = ")
    # print(Fdiff)


