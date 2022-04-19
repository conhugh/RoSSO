# Optimization of the performance of stochastic surveillance strategies
import numpy as np
import os
import time
from collections import deque
import math
import jax
import optax
from jax import grad, jit, devices
import jax.numpy as jnp
from StratCompJax import *
from StratViz import *

# Initialize parameters of the model + optimizer.
def test_optimizer(P0, F0, tau, cols, eps0, iters):
    P = P0
    n = P.shape[0]

    start_learning_rate = eps0
    optimizer = optax.sgd(start_learning_rate, momentum=0.99, nesterov=True)

    # params = {'P': P0, 'F0': F0, 'tau': tau}
    opt_state = optimizer.init(P)
    for k in range(iters + 1):
        # print("iteration number: " + str(k))
        # grads = compMCPGradJIT(params['P'], params['F0'], params['tau']
        # start_time = time.time()
        grads = compMCPGradJIT(P, F0, tau).block_until_ready()
        grads = zeroGradColsJIT(grads, cols).block_until_ready()
        grads = -1*grads  # negate so that the optimizer does gradient ascent
        # print("--- Getting grad took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        grads = grads.reshape((n, n), order='F').block_until_ready()
        # print("--- Reshaping grad took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        updates, opt_state = optimizer.update(grads, opt_state)
        # print("--- Getting update and state took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        P = optax.apply_updates(P, updates).block_until_ready()
        # print("--- Applying update took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        P = projOntoSimplexJIT(P).block_until_ready()
        # print("--- Simplex projection took: %s seconds ---" % (time.time() - start_time))
        if k % 100 == 0:
            print("MCP at iteration " + str(k) + ":")
            print(jnp.min(computeCapProbsJIT(P, F0, tau)))
            print("P at iteration " + str(k) + ":")
            print(P)
    print("Initial P = ")
    print(P0)
    print("For eps0 = " + str(eps0) + ", P at iteration " + str(k) + ":")
    print(P)
    F = computeCapProbsJIT(P, F0, tau).block_until_ready()
    print("Minimum Capture Probability at iteration " + str(iter) + ":")
    print(jnp.min(F))
    return P, F


# Initialize parameters of the model + optimizer.
def sgd_Pconv(P0, F0, tau, cols, eps0, radius):
    n = P0.shape[0]
    P = P0 
    avgP = P  # running avg capture probability matrix, for checking convergence

    start_learning_rate = eps0
    optimizer = optax.sgd(start_learning_rate, momentum=0.99, nesterov=True)
    opt_state = optimizer.init(P)

    iter = 0  # number of gradient ascent steps taken so far
    converged = False
    while not converged:
        # print("iteration number: " + str(iter))
        # take gradient ascent step:
        grads = compMCPGradJIT(P, F0, tau).block_until_ready()
        grads = zeroGradColsJIT(grads, cols).block_until_ready()
        grads = -1*grads  # negate so that the optimizer does gradient ascent
        # print("--- Getting grad took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        grads = grads.reshape((n, n), order='F').block_until_ready()
        # print("--- Reshaping grad took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        updates, opt_state = optimizer.update(grads, opt_state)
        # print("--- Getting update and state took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        P = optax.apply_updates(P, updates).block_until_ready()
        # print("--- Applying update took: %s seconds ---" % (time.time() - start_time))
        # start_time = time.time()
        P = projOntoSimplexJIT(P).block_until_ready()
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
    print("Initial P = ")
    print(P0)
    print("For eps0 = " + str(eps0) + ", P at iteration " + str(iter) + ":")
    print(P)
    F = computeCapProbsJIT(P, F0, tau).block_until_ready()
    print("Minimum Capture Probability at iteration " + str(iter) + ":")
    print(jnp.min(F))
    return P, F


# Initialize parameters of the model + optimizer.
def sgd_MAPconv(P0, F0, tau, cols, eps0, radius, numRecPs, maxIters):
    n = P0.shape[0]
    P = P0 

    optimizer = optax.sgd(eps0, momentum=0.99, nesterov=True)
    opt_state = optimizer.init(P)

    recentPs = deque()  # queue storing recent P matrices, for checking convergence
    recentPs.append(P)  
    # compute moving average of recent P matrices before adding new one:
    oldMAP = np.mean(recentPs, axis=0) 

    iter = 0  # number of gradient ascent steps taken so far
    converged = False
    while not converged:
        # print("iteration number: " + str(iter))
        # take gradient ascent step:
        grads = compMCPGradJIT(P, F0, tau).block_until_ready()
        grads = zeroGradColsJIT(grads, cols).block_until_ready()
        grads = -1*grads  # negate so that the optimizer does gradient ascent
        grads = grads.reshape((n, n), order='F').block_until_ready()
        updates, opt_state = optimizer.update(grads, opt_state)
        P = optax.apply_updates(P, updates).block_until_ready()
        P = projOntoSimplexJIT(P).block_until_ready()
        # print status info to terminal:
        # if (iter % 100) == 0:
        #     F = computeCapProbsJIT(P, F0, tau).block_until_ready()
        #     print("Minimum Capture Probability at iteration " + str(iter) + ":")
        #     print(jnp.min(F))

        # check for convergence, update running avg P matrices and step counter:
        # start_time = time.time()
        # update queue of recent P matrices:
        recentPs.append(P)
        # compute updated moving average of P matrices:
        if iter > numRecPs:
            oldestP = recentPs.popleft() 
            newMAP = (oldMAP*numRecPs - oldestP + P)/numRecPs
        else:
            newMAP = np.mean(recentPs, axis=0)
        
        # compute change in moving average of P matrices:
        diffMAP = np.abs(newMAP - oldMAP)
        # check for element-wise convergence, update moving average P matrix and iteration counter:
        converged = jnp.max(diffMAP) < radius
        oldMAP = newMAP
        if iter == maxIters:
            converged = True
        iter = iter + 1
    # print("Initial P = ")
    # print(P0)
    # print("For eps0 = " + str(eps0) + ", P at iteration " + str(iter) + ":")
    # print(P)
    F = computeCapProbsJIT(P, F0, tau).block_until_ready()
    print("Minimum Capture Probability at iteration " + str(iter) + ":")
    print(jnp.min(F))
    return P, F, iter

# Explore optima for the given graph and attack duration:
def exploreGraphOptima(A, tau, testSetNum, graphNum, numInitPs):
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    cols = getZeroCols(A)
    initPs = initRandPs(A, numInitPs)

    initSteps = np.zeros(numInitPs, dtype='float32')
    stepScales = np.zeros(numInitPs, dtype='float32')
    convIters = np.zeros(numInitPs, dtype='float32')
    convTimes = np.zeros(numInitPs, dtype='float32')

    rad = 0.00001
    numRecPs = 100
    maxIters = 10000

    # Create directory for saving the optimization results:
    project_dir = os.getcwd()
    results_dir = os.path.join(project_dir, "Results/test_set_" + str(testSetNum))

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
    graph_name = "graph" + str(graphNum) + "_tau" + str(tau)
    graph_dir = os.path.join(results_dir, graph_name)
    if not os.path.isdir(graph_dir):
        os.mkdir(graph_dir)

    # Save the env graph binary adjacency matrix:
    bin_adj_path = os.path.join(graph_dir, "A.csv")
    np.savetxt(bin_adj_path, A, delimiter=',')
    # Save a drawing of the env graph:
    drawEnvGraph(A, graphNum, graph_dir)

    # Run the optimization algorithm:
    for k in range(numInitPs):
        print("Optimizing with initial P matrix number " + str(k + 1) + "...")
        P0 = initPs[:, :, k]
        initMCPGrad = compMCPGradJIT(P0, F0, tau)  
        initMCPGradZ = zeroGradColsJIT(initMCPGrad, cols)
        stepScale = np.max(initMCPGradZ)
        stepScales[k] = stepScale
        eps0 = 0.0001/stepScale
        initSteps[k] = eps0
        start_time = time.time()
        P, F, iters = sgd_MAPconv(P0, F0, tau, cols, eps0, rad, numRecPs, maxIters)
        convIters[k] = iters
        convTime = time.time() - start_time
        convTimes[k] = convTime
        print("--- Convergence took: %s seconds ---" % (convTime))
        # Save initial and final P matrices:
        init_P_name = "init_P_" + str(k + 1) + ".csv"
        init_P_path = os.path.join(graph_dir, init_P_name)
        np.savetxt(init_P_path, P0, delimiter=',')
        opt_P_name = "opt_P_" + str(k + 1) + ".csv"
        opt_P_path = os.path.join(graph_dir, opt_P_name)
        np.savetxt(opt_P_path, P, delimiter=',')


    # Save information about graph and the optimization algorithm used:
    info_path = os.path.join(graph_dir, "info.txt")
    with open(info_path, 'w') as info:
        info.write("Graph Information:\n")
        info.write("Number of nodes (n) = " + str(n) + "\n")
        info.write("Attack Duration (tau) = " + str(tau) + "\n")
        info.write("Optimizer Information:\n")
        info.write("Optimizer used = sgd_MAPconv\n")
        info.write("Convergence radius = " + str(rad) + "\n")
        info.write("Moving average window size (numRecPs) = " + str(numRecPs) + "\n")
        info.write("Max allowed number of iterations (maxIters) = " + str(maxIters) + "\n")
        info.write("Initial Step Sizes (initSteps) = " + str(initSteps) + "\n")
        info.write("Max element of initial MCP gradients (stepScales) = " + str(stepScales) + "\n")
        info.write("Number of iterations required = " + str(convIters) + "\n")
        info.write("Optimization Times Required (seconds) = " + str(convTimes) + "\n")
    info.close()

# TESTING ------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    # A = genStarG(6)
    # A = genLineG(6)

    # A1 = jnp.array([[1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 1]])  
    # A2 = genSplitStarG(2, 3, 2)
    # # print(A2)
    
    leftLeavesR = np.arange(3, 6)
    rightLeavesR = np.arange(2, 5)
    midLineLenR = np.arange(2, 5)

    testSetNum = 1
    project_dir = os.getcwd()
    results_dir = os.path.join(project_dir, "Results/test_set_" + str(testSetNum))
    if os.path.isdir(results_dir):
        input("WARNING! The test set directory already exists, press enter to continue and overwrite data.")

    graphNum = 1
    numInitPs = 10
    for i in range(len(midLineLenR)):
        for j in range(len(rightLeavesR)):
            for k in range(len(leftLeavesR)):
    # for i in range(1):
    #     for j in range(1):
    #         for k in range(2):
                print("-------- Working on Graph Number " + str(graphNum) + "----------")
                A = genSplitStarG(leftLeavesR[k], rightLeavesR[j], midLineLenR[i])
                tau1 = midLineLenR[i] + 1 
                tau2 = midLineLenR[i] + 3
                exploreGraphOptima(A, tau1, testSetNum, graphNum, numInitPs)
                exploreGraphOptima(A, tau2, testSetNum, graphNum, numInitPs)
                graphNum = graphNum + 1





    