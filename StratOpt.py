# Optimization of the performance of stochastic surveillance strategies
import numpy as np
import time
from collections import deque
np.set_printoptions(linewidth=np.inf)
import math
from StratComp import *
from StratViz import *

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
    # take each of the unconstrained gradient steps under consideration:
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

# Apply constraints to gradient ascent step suggested by avg of gradients of lowest cap probs, 
# take the step and return the corresponding updated transition probability matrix P
def avgGradAscentStep(P, J, F, tau, eps, lcpNum):
    n = P.shape[0]
    Fvec = F.flatten('F')
    # get indices of lcpNum lowest capture probabilities in cap prob vector
    lcpLocs = np.argpartition(Fvec, lcpNum)[0:lcpNum]
    # print("lcpLocs = ")
    # print(lcpLocs)
    # print("Fvec at lcpLocs = ")
    # print(Fvec[lcpLocs])
    # print("Fvec = ")
    # print(Fvec)
    # wait = input("Press enter to continue...")
    lcpNum = lcpLocs.shape[0]
    # compute corresponding average unconstrained gradient ascent step
    avgUGradStep = np.zeros([n**2])
    for i in range(lcpNum):
        avgUGradStep = avgUGradStep + eps*J[lcpLocs[i], :].reshape(n**2)
    avgUGradStep = avgUGradStep/lcpNum
    # generate new Pvec and P matrix resulting from the avg gradient ascent step
    newPvec = P.flatten('F') + avgUGradStep
    newPmat = newPvec.reshape((n, n), order='F') 
    newPmat = projOntoSimplex(newPmat)
    # print("newPmat = ")
    # print(newPmat)
    # wait = input("Press enter to continue...")
    newPF = np.full([n, n, 2], np.nan)
    # return new P matrix and cap prob mat corresponding to (an) optimal step choice
    newPF[:, :, 0] = newPmat
    newPF[:, : , 1] = computeCapProbs(newPmat, tau)
    return newPF

# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscentFixed(P0, A, tau, eps, iterations):
    P = P0
    F = computeCapProbs(P, tau)
    for k in range(iterations):
        J = compCPJac(P, tau)
        J = zeroCPJacCols(J, A)
        newPF = gradAscentStep(P, J, F, tau, eps)
        P = newPF[:, :, 0]
        F = newPF[:, :, 1]
        if k % 20 == 0:
            print("Minimum Capture Probability at iteration " + str(k) + ":")
            print(np.min(F))
    return P, F

# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscentFConv(P0, A, tau, eps0, radius):
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
        if (iter % 500) == 0:
            print("Minimum Capture Probability at iteration " + str(iter) + ":")
            print(np.min(F))
            # print("F at iteration " + str(iter) + ":")
            # print(F)
        # check for convergence, update running avg cap probs and step counter:
        newAvgF = ((iter)*avgF + F)/(iter + 1)
        diffAvgF = np.abs(newAvgF - avgF)
        converged = np.amax(diffAvgF) < radius
        avgF = newAvgF
        iter = iter + 1
    print("Minimum Capture Probability at iteration " + str(iter - 1) + ":")
    print(np.min(F))
    print("Final diffAvgF = ")
    print(diffAvgF)
    return P, F

# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscentPConv(P0, A, tau, eps0, radius):
    epsThresh = 0.001  # threshold (i.e., minimum) step size to be used
    thresh = 1000  # step number at which step size reaches threshold value
    iter = 0  # number of gradient ascent steps taken so far
    P = P0 
    F = computeCapProbs(P, tau)
    avgP = P  # running avg capture probability matrix, for checking convergence
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
        # newPF = avgGradAscentStep(P, J, F, tau, eps, 3)  # [AVERAGE GRADIENT OF LOWEST CAP PROBS]
        newPF = gradAscentStep(P, J, F, tau, eps)
        P = newPF[:, :, 0]
        F = newPF[:, :, 1]
        # print("P = ")
        # print(P)
        # print("F = ")
        # print(F)
        # wait = input("Press enter to continue...")
        # print status info to terminal:
        if (iter % 500) == 0:
            print("Minimum Capture Probability at iteration " + str(iter) + ":")
            print(np.min(F))
            # print("F at iteration " + str(iter) + ":")
            # print(F)
        # check for convergence, update running avg cap probs and step counter:
        newAvgP = ((iter)*avgP + P)/(iter + 1)
        diffAvgP = np.abs(newAvgP - avgP)
        converged = np.amax(diffAvgP) < radius
        avgF = newAvgP
        iter = iter + 1
    print("Minimum Capture Probability at iteration " + str(iter - 1) + ":")
    print(np.min(F))
    # print("Final diffAvgF = ")
    # print(diffAvgP)
    return P, F

# Perform gradient ascent to increase min cap prob starting from initial trans prob mat P0
def gradAscentMCPConv(P0, A, tau, eps0, radius):
    epsThresh = 0.001  # threshold (i.e., minimum) step size to be used
    thresh = 1000  # step number at which step size reaches threshold value
    iter = 0  # number of gradient ascent steps taken so far
    P = P0 
    F = computeCapProbs(P, tau)
    numRecMCPs = 100  # number of recent min cap probs to use when computing moving average 
    recentMCPs = deque()  # queue storing recent min cap probs, for checking convergence
    recentMCPs.append(np.min(F))  
    # compute moving average of recent min cap probs before adding new one:
    oldMAMCP = np.mean(recentMCPs) 
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
        if (iter % 500) == 0:
            print("Minimum Capture Probability at iteration " + str(iter) + ":")
            print(np.min(F))
            print("Moving Avg Min Cap Prob = ")
            print(oldMAMCP)
        # get new min cap prob:
        newMCP = np.min(F)  
        # update queue of recent min cap probs:
        recentMCPs.append(newMCP)
        if iter > numRecMCPs:
            recentMCPs.popleft()
        # compute updated moving average of min cap probs:
        newMAMCP = np.mean(recentMCPs)
        # compute change in moving average of min cap probs:
        diffMAMCP = np.abs(newMAMCP - oldMAMCP)
        # check for convergence, update moving average mcp and iteration counter:
        converged = diffMAMCP < radius
        oldMAMCP = newMAMCP
        iter = iter + 1
    print("Minimum Capture Probability at iteration " + str(iter - 1) + ":")
    print(np.min(F))
    return P, F

# NOT WORKING YET, START HERE:
def RMSProp(P0, A, tau, eps, rho, radius):
    n = P0.shape[0]
    r = 0
    delta = 10**(-6)
    iter = 0  # number of gradient ascent steps taken so far
    P = P0 
    avgP = P  # running avg capture probability matrix, for checking convergence
    converged = False
    while not converged:
        # get gradients:
        J = compCPJac(P, tau)
        J = zeroCPJacCols(J, A)
        F = computeCapProbs(P, tau)
        Fvec = F.flatten('F')
        # get indices of minimal capture probabilities in cap prob vector
        minCapProb = np.min(Fvec)
        mcpLocs = np.argwhere(Fvec == minCapProb)
        mcpNum = mcpLocs.shape[0]
        # compute average of gradients of min cap probs:
        grads = np.zeros([n**2, mcpNum])
        for i in range(mcpNum):
            grads[:, i] = J[mcpLocs[i], :].reshape(n**2)
        uGradStep = np.mean(grads, 1)

        r = rho*r + (1 - rho)*(uGradStep*uGradStep)  # accumulate squared gradients
        deltaP = -eps/(delta + r)*uGradStep  # compute gradient step
        newPvec = P.flatten('F') + deltaP # compute new Pvec
        newPmat = newPvec.reshape((n, n), order='F') 
        newPmat = projOntoSimplex(newPmat)
        # print("r = ")
        # print(r)
        # print("deltaP = ")
        # print(deltaP)
        # print("newPmat = ")
        # print(newPmat)
        # wait = input("Press enter to continue...")
        newPF = np.full([n, n, 2], np.nan)
        # return new P matrix and cap prob mat corresponding to (an) optimal step choice
        newPF[:, :, 0] = newPmat
        newPF[:, : , 1] = computeCapProbs(newPmat, tau)
        # print status info to terminal:
        if (iter % 10) == 0:
            print("Minimum Capture Probability at iteration " + str(iter) + ":")
            print(np.min(F))
            # print("F at iteration " + str(iter) + ":")
            # print(F)
        # check for convergence, update running avg cap probs and step counter:
        newAvgP = ((iter)*avgP + P)/(iter + 1)
        diffAvgP = np.abs(newAvgP - avgP)
        converged = np.amax(diffAvgP) < radius
        avgF = newAvgP
        iter = iter + 1
    print("Minimum Capture Probability at iteration " + str(iter - 1) + ":")
    print(np.min(F))
    # print("Final diffAvgF = ")
    # print(diffAvgP)
    return P, F


def exploreOptima(A, tau, trials):
    n = A.shape[0]
    initPMats = np.full([n, n, trials], np.nan)
    optPMats = np.full([n, n, trials], np.nan)
    minCapProbs = np.full([trials], np.nan)
    for i in range(trials):
        P0 = initRandP(A)
        initPMats[:, :, i] = P0
        [P, F] = gradAscentPConv(P0, A, tau, 0.05, 0.00001)
        optPMats[:, :, i] = P
        minCapProbs[i] = np.min(F)
    return initPMats, optPMats, minCapProbs


# TESTING ------------------------------------------------------------------------
# np.set_printoptions(suppress=True)
# # A = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]])
# # A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
# # A = np.array([[0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
# A = np.array([[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]])
# # A = np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1]])
P0 = initRandP(A)
# # # P0 = np.array([[0, 0.2, 1 - 0.2], [1, 0, 0], [1, 0, 0]])
tau = 3
# # trials = 10

# [P, F] = gradAscentPConv(P0, A, tau, 0.05, 0.00001)
# print("P0 = ")
# print(P0)
# print("P_final = ")
# print(P)


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


# # [P, F] = gradAscentFixed(P0, A, tau, 0.05, 1000)
# # [P, F] = gradAscentMCPConv(P0, A, tau, 0.05, 0.0000005)
# # [P, F] = gradAscentFConv(P0, A, tau, 0.05, 0.00001)
start_time = time.time()
# [P, F] = gradAscentPConv(P0, A, tau, 0.05, 0.00001)
[P, F] = RMSProp(P0, A, tau, 0.0001, 0.9, 0.000001)

print("--- Optimization took: %s seconds ---" % (time.time() - start_time))
print("P0 = ")
print(P0)
print("P_final = ")
print(P)

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

# STRAT_OPT_PARAMS = {
#     "first": "first entry",
#     "sec": "second entry"
# }

# print(STRAT_OPT_PARAMS)
# print(STRAT_OPT_PARAMS["sec"])
# print(STRAT_OPT_PARAMS["first"])

# K = 2
# test = np.array([5.1, 3.1, 2.1, 75, 32, 1.8, 3.4, 2.9, 89, 102, 15, 12, 8.4])
# testK = np.argpartition(test, K)
# print(testK)
# print(test[testK[0:K]])