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

# Initialize parameters of the model + optimizer.
# def sgd_GradConv(P0, F0, tau, cols, eps0, radius, numRecPdiffs, maxIters, gradMode="mcp", lcpNum=1):
def sgd_GradConv(P0, F0, tau, cols, opt_params):
    n = P0.shape[0]
    P = P0 
    optimizer = optax.sgd(opt_params["eps0"], momentum=0.99, nesterov=True)
    opt_state = optimizer.init(P)
    recentPdiffs = deque()  # queue storing recent P matrices, for checking convergence

    ubPupdate = jnp.full((n, n), 0.01)
    lbPupdate = jnp.full((n, n), -0.01)
    trackedVals = {
        "iters" : [],
        "PdiffSums" : [],
        "PdiffMaxElts" : [],
        "mcpInds" : [],
        "mcps" : [],
        "finalMCP" : [],
        "finalIters" : []
    }

    iter = 0  # number of gradient ascent steps taken so far
    converged = False
    while not converged:
        # print("iteration number: " + str(iter))
        # take gradient ascent step:
        if opt_params["gradMode"] == "mcp":
            grad = compMCPGradJIT(P, F0, tau).block_until_ready()
        else:
            grad = compAvgLCPGradJIT(P, F0, tau, opt_params["lcpNum"]).block_until_ready()
        
        grad = zeroGradColsJIT(grad, cols).block_until_ready()
        grad = -1*grad  # negate so that the optimizer does gradient ascent
        grad = grad.reshape((n, n), order='F').block_until_ready()
        if(iter % 200 == 0):
            print("------ iteration number " + str(iter) + " -------")
            print("grad 1-norm = ")
            print(jnp.sum(grad))
            print("grad largest element = ")
            print(jnp.min(grad))
        
        updates, opt_state = optimizer.update(grad, opt_state)
        # bound the update to the P matrix:
        updates = jnp.minimum(updates, ubPupdate)
        updates = jnp.maximum(updates, lbPupdate)

        # apply update to P matrix:
        oldP = P
        P = optax.apply_updates(P, updates).block_until_ready()
        P = projOntoSimplexJIT(P).block_until_ready()

        Pdiff = P - oldP
        absPdiffSum = jnp.sum(jnp.abs(Pdiff))

        # check for convergence, update running avg absPdiffSum and step counter:
        # update queue of recent P matrices:
        recentPdiffs.append(absPdiffSum)
        # compute updated moving average of P matrices:
        if iter > opt_params["numRecPdiffs"]:
            oldestPdiff = recentPdiffs.popleft() 
            newMAPdiff = (oldMAPdiff*opt_params["numRecPdiffs"] - oldestPdiff + absPdiffSum)/opt_params["numRecPdiffs"]
        else:
            newMAPdiff = np.mean(recentPdiffs)
        
        # track metrics of interest:
        if(iter % 10 == 0):
            trackedVals["iters"].append(iter)
            trackedVals["PdiffSums"].append(jnp.sum(jnp.abs(Pdiff)))
            if(jnp.abs(jnp.max(Pdiff)) > jnp.abs(jnp.min(Pdiff))):
                trackedVals["PdiffMaxElts"].append(jnp.max(Pdiff))
            else:
                trackedVals["PdiffMaxElts"].append(jnp.min(Pdiff))                
            F = computeCapProbsJIT(P, F0, tau).block_until_ready()
            F = F.reshape((n**2), order='F')
            trackedVals["mcpInds"].append(jnp.argmin(F))
            trackedVals["mcps"].append(jnp.min(F))

        # if(iter % 2000 == 0):
        #     # cut the step size in half
        #     eps0 = eps0/2
        #     optimizer = optax.sgd(eps0, momentum=0.99, nesterov=True)
        #     opt_state = optimizer.init(P)

        # check for element-wise convergence, update moving average absPdiffSum and iteration counter:
        if(iter > opt_params["numRecPdiffs"]):
            converged = newMAPdiff < opt_params["radius"]
        oldMAPdiff = newMAPdiff
        if iter == opt_params["maxIters"]:
            converged = True
        iter = iter + 1
    F = computeCapProbsJIT(P, F0, tau).block_until_ready()
    finalMCP = jnp.min(F)
    trackedVals["finalMCP"].append(finalMCP)
    print("Minimum Capture Probability at iteration " + str(iter) + ":")
    print(finalMCP)
    trackedVals["finalIters"].append(iter)

    return P, F, trackedVals

# Explore optima for the given graph and attack duration:
def exploreGraphOptima(A, tau, testSetName, graphNum, numInitPs, gradMode):
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    cols = getZeroCols(A)
    initPs = initRandPs(A, numInitPs)

    initSteps = np.zeros(numInitPs, dtype='float32')
    stepScales = np.zeros(numInitPs, dtype='float32')
    convIters = np.zeros(numInitPs, dtype='float32')
    convTimes = np.zeros(numInitPs, dtype='float32')
    finalMCPs = np.zeros(numInitPs, dtype='float32')

    opt_params = {
        "radius" : 0.00001,
        "numRecPdiffs" : 200,
        "eps0" : None,
        "maxIters" : 20000,
        "gradMode" : gradMode,
        "lcpNum" : int(np.ceil((n**2)/10)),
    }

    print("lcpNum = " + str(opt_params["lcpNum"]))

    # Create directory for saving theb results for the given graph:
    project_dir = os.getcwd()
    results_dir = os.path.join(project_dir, "Results/test_set_" + str(testSetName))
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    graph_name = "graph" + str(graphNum) + "_tau" + str(tau)
    graph_dir = os.path.join(results_dir, graph_name)
    if not os.path.isdir(graph_dir):
        os.mkdir(graph_dir)

    # Run the optimization algorithm:
    for k in range(numInitPs):
        print("Optimizing with initial P matrix number " + str(k + 1) + "...")
        P0 = initPs[:, :, k]
        if gradMode == "mcp":
            initGrad = compMCPGradJIT(P0, F0, tau)  
        else:
            initGrad = compAvgLCPGradJIT(P0, F0, tau, opt_params["lcpNum"])  
        initGradZ = zeroGradColsJIT(initGrad, cols)

        # TO BE REPLACED BY setStepSize function: ----------------------------
        stepScale = np.max(initGradZ)
        stepScales[k] = stepScale
        eps0 = 0.01/stepScale
        opt_params["eps0"] = eps0
        initSteps[k] = eps0
        # --------------------------------------------------------------------
        start_time = time.time()
        # P, F, trackedVals = sgd_GradConv(P0, F0, tau, cols, eps0, rad, numRecPs, maxIters, gradMode)
        P, F, trackedVals = sgd_GradConv(P0, F0, tau, cols, opt_params)
        convIters[k] = trackedVals["finalIters"][0]
        convTime = time.time() - start_time
        convTimes[k] = convTime
        finalMCPs[k] = trackedVals["finalMCP"][0]
        print("--- Optimization took: %s seconds ---" % (convTime))
        # Save initial and optimized P matrices:
        np.savetxt(graph_dir + "/init_P_" + str(k + 1) + ".csv", P0, delimiter=',')
        np.savetxt(graph_dir + "/opt_P_" + str(k + 1) + ".csv", P, delimiter=',')
        metrics_dir = os.path.join(graph_dir, "metrics")
        if not os.path.isdir(metrics_dir):
            os.mkdir(metrics_dir)
        for metric in trackedVals.keys():
            if metric.find("final") == -1:
                np.savetxt(metrics_dir + "/" + metric + "_" + str(k + 1) + ".csv", trackedVals[metric], delimiter=',')

    # Save the env graph binary adjacency matrix:
    np.savetxt(graph_dir + "/A.csv", A, delimiter=',')
    # Save a drawing of the env graph:
    drawEnvGraph(A, graphNum, graph_dir)
    # Generate unique graph code:
    graph_code = genGraphCode(A)
    # Write info file with graph and optimization algorithm info:
    info_path = os.path.join(graph_dir, "info.txt")
    with open(info_path, 'w') as info:
        info.write("Graph Information:\n")
        info.write("Number of nodes (n) = " + str(n) + "\n")
        info.write("Attack Duration (tau) = " + str(tau) + "\n")
        info.write("Graph Code = " + graph_code + "\n")
        info.write("Optimizer Information:\n")
        info.write("Optimizer used = sgd_GradConv\n")
        info.write("Gradient Mode = " + opt_params["gradMode"] + "\n")
        info.write("Number of LCPs used, if applicable = " + str(opt_params["lcpNum"]) + "\n")
        info.write("Convergence radius = " + str(opt_params["radius"]) + "\n")
        info.write("Moving average window size (numRecPs) = " + str(opt_params["numRecPdiffs"]) + "\n")
        info.write("Max allowed number of iterations (maxIters) = " + str(opt_params["maxIters"]) + "\n")
        info.write("Initial Step Sizes (initSteps) = " + str(initSteps) + "\n")
        info.write("Max element of initial MCP gradients (stepScales) = " + str(stepScales) + "\n")
        info.write("Final MCP achieved = " + str(finalMCPs) + "\n")
        info.write("Number of iterations required = " + str(convIters) + "\n")
        info.write("Optimization Times Required (seconds) = " + str(convTimes) + "\n")
    info.close()


def setStepSize(A, tau, eps0, numInitPs):
    absGradSums = jnp.zeros(numInitPs)
    absGradMaxs = jnp.zeros(numInitPs)
    cols = getZeroCols(A)
    initPs = initRandPs(A, numInitPs)
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    for k in range(numInitPs):
        P0 = initPs[:, :, k]
        initMCPGrad = compMCPGradJIT(P0, F0, tau)  
        initMCPGradZ = zeroGradColsJIT(initMCPGrad, cols)
        absGradSums = absGradSums.at[k].set(jnp.sum(jnp.abs(initMCPGradZ)))
        absGradMaxs = absGradMaxs.at[k].set(jnp.max(jnp.abs(initMCPGradZ)))
    meanGradSum = jnp.mean(absGradSums)
    meanGradMax = jnp.mean(absGradMaxs)
    # stepSize = eps0 + 10*(0.1 - meanGradSum)
    # stepSize = eps0 - 0.01*(jnp.log10(meanGradSum/10))
    stepSize = -eps0/jnp.log10(meanGradSum)
    return stepSize, meanGradSum, meanGradMax, absGradMaxs


# TESTING ------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    # leftLeavesR = np.arange(3, 6)
    # rightLeavesR = np.arange(2, 5)
    # midLineLenR = np.arange(2, 5)
    leftLeavesR = np.array([5])
    rightLeavesR = np.array([4])
    midLineLenR = np.array([4])

    # A = genSplitStarG(5, 4, 4)
    # tau = 5
    # eps0 = 0.001 
    # numInitPs = 30
    # eps, mgs, mgm, agms = setStepSize(A, tau, eps0, numInitPs)
    # print("numInitPs = " + str(numInitPs))
    # print("step = " + str(eps))
    # print("mgs = " + str(mgs))
    # print("mgm = " + str(mgm))
    # # print("agms = ")
    # # print(agms)
    # print("max agm =" + str(jnp.max(agms)) + ", min agm = " + str(jnp.min(agms)))
    # agmRatio = jnp.max(agms)/jnp.min(agms)
    # print("max/min agm ratio = " + str(agmRatio))

    testSetName = "StratVizTesting"
    project_dir = os.getcwd()
    results_dir = os.path.join(project_dir, "Results/test_set_" + str(testSetName))
    if os.path.isdir(results_dir):
        input("WARNING! The test set directory already exists, press enter to continue and overwrite data.")

    graphNum = 1
    numInitPs = 3
    # for i in range(len(midLineLenR)):
    #     for j in range(len(rightLeavesR)):
    #         for k in range(len(leftLeavesR)):
    # for i in range(2, 3):
    #     for j in range(2, 3):
    #         for k in range(2, 3):
    for i in range(1):
        for j in range(1):
            for k in range(2, 4):
                print("-------- Working on Graph Number " + str(graphNum) + "----------")
                # A = genSplitStarG(leftLeavesR[k], rightLeavesR[j], midLineLenR[i])
                # A = genStarG(6)
                A = genGridG(3, k)
                tau1 = k + 1
                tau2 = k + 3
                # tau1 = 2
                # tau2 = 4
                # tau1 = midLineLenR[i] + 1 
                # tau2 = midLineLenR[i] + 3
                exploreGraphOptima(A, tau1, testSetName, graphNum, numInitPs, gradMode="mcp")
                exploreGraphOptima(A, tau2, testSetName, graphNum, numInitPs, gradMode="mcp")
                graphNum = graphNum+ 1





    