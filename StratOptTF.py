# Optimization of the performance of stochastic surveillance strategies using automatic differentiation
import time
import random
import numpy as np
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # FIX LATER
sys.path.insert(0, 'C:/Users/conno/anaconda3/envs/research_env/Lib/site-packages/')
# print("sys path = " + str(sys.path))
# print("os path dirname = " + str(os.path.dirname))
import tensorflow as tf
# from StratComp import *

# class CompMCP(tf.Module):
#   def __init__(self, P0, tau, name=None):
#     super(CompMCP, self).__init__(name=name)
#     self.P = P0
#     self.tau = tau
#     self.F = computeCapProbs(P0, tau)
#     self.mcp = np.min(self.F)
#   def __call__(self):
#     self.F = computeCapProbs(self.P, self.tau)
#     mcp = np.min(self.F)
#     return mcp

# class CompMCP(tf.Module):
#   def __init__(self, P0, tau, name=None):
#     super(CompMCP, self).__init__(name=name)
#     self.P = P0
#     self.tau = tau
#     self.F = computeCapProbs(P0, tau)
#     self.mcp = np.min(self.F)
#   def __call__(self):
#     self.F = computeCapProbs(self.P, self.tau)
#     mcp = np.min(self.F)
#     return mcp

# A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
# P0 = initRandP(A)
# tau = 3
# MCP_computation = CompMCP(P0, tau) 
    
# def gradAscentTF(P0, tau, eps, iterations):
#     for i in range(iterations):
#         with tf.GradientTape() as t:
#             tau = tf.stop_gradient(MCP_computation.tau)
#             mcp = MCP_computation()
#         dP = t.gradient(mcp, MCP_computation.P)
#         MCP_computation.P = MCP_computation.P + eps*dP
#         MCP_computation.P = projOntoSimplex(MCP_computation.P)
#         if i % 100 == 0:
#             print("Minimum Capture Probability at iteration " + str(i) + ":")
#             print(MCP_computation.mcp)
#     print("Final Min Cap Prob: ")
#     print(MCP_computation.mcp)
#     print("Final P: ")
#     print(MCP_computation.P)
    # return MCP_computation.P, MCP_computation.F


# Takes:   A, the binary adjacency matrix corresponding to the environment graph
# Returns: P0, a random transition probability matrix which is valid (i.e. row-stochastic) 
#              and consistent with the environment graph described by A (see XD-DP-FB_19b.pdf)
def initRandP(A):
    random.seed(1)
    P0 = np.zeros_like(A, dtype='float64')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P0[i, j] = random.random()
    P0 = np.matmul(np.diag(1/np.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    # P0 = tf.convert_to_tensor(P0)
    return P0

# def initRandP(A):
#     # random.seed(2)
#     n = tf.shape(A)[0]
#     P0 = tf.zeros_like(A)
#     for i in range(n):
#         for j in range(n):
#             if A[i, j] != 0:
#                 P0[i, j] = tf.Variable(random.random())
#     P0 = tf.linalg.matmul(tf.linalg.diag(1/(tf.reduce_sum(P0, axis=1))))   # normalize to generate valid prob dist
#     return P0

# Compute first hitting time probability matrices (up to tau time steps) for fixed P matrix 
def computeFHTProbMats(P, tau):  #DEAL WITH TF ITEM ASSIGNMENT ISSUE
    # print("tau = " + str(tau))
    n = tf.shape(P)[0]
    F = tf.zeros([n, n, tau], dtype='float64')
    F[:, :, 0] = P
    for i in range(1, tau):
        F[:, :, i] = tf.linalg.matmul(P, (F[:, :, i - 1] - tf.linalg.diag(tf.linalg.diag_part(F[:, :, i - 1], k=0, padding_value=0), k=0, num_rows=n, num_cols=n, padding_value=0)))
    return F

# Compute capture probabilities for each pair of nodes for fixed P matrix
def computeCapProbs(P, tau):
    F = computeFHTProbMats(P, tau)
    capProbs = tf.math.reduce_sum(F, axis=2)
    return capProbs

# Project the given trans prob vector onto nearest point on probability simplex, if applicable
# See https://arxiv.org/abs/1309.1541 for explanation of the algorithm used here
def projOntoSimplex(P): 
    n = tf.shape(P)[0]
    sortMapping = tf.argsort(P, axis=1, direction='DESCENDING')
    invSortMapping = tf.argsort(sortMapping, axis=1, direction='ASCENDING')
    rowInds = tf.Variable([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5]])
    mapping = tf.stack([rowInds, sortMapping], axis=2)
    invMapping = tf.stack([rowInds, invSortMapping], axis=2)

    X = tf.gather_nd(P, mapping)
    Xtmp = tf.linalg.matmul(tf.math.cumsum(X, axis=1) - 1, tf.linalg.diag(1/tf.range(1, n + 1)))

    rhoVals = tf.reduce_sum(tf.cast(tf.greater(X, Xtmp), dtype='int32'), axis=1) - 1 
    indices = tf.stack([tf.range(n), rhoVals], axis=1)
    lambdaVals = -1*tf.gather_nd(Xtmp, indices)

    newX = tf.math.maximum(X + tf.transpose(lambdaVals*tf.ones([n, n], dtype='float64')), tf.zeros([n, n], dtype='float64'))
    newP = tf.gather_nd(newX, invMapping)

    return newP

# TO DO: rewrite CP computation with tf variables
def gradAscentTF(P0, tau0, eps, iterations):
    P = tf.Variable(P0)
    tau = tf.Variable(tau0)
    for i in range(iterations):
        with tf.GradientTape() as t:
            tau = tf.stop_gradient(tau)
            F = computeCapProbs(P, tau)
            mcp = tf.math.reduce_min(F)
        dP = t.gradient(mcp, P)
        P = P + eps*dP
        P = projOntoSimplex(P)
        if i % 100 == 0:
            print("Minimum Capture Probability at iteration " + str(i) + ":")
            print(mcp)
    F = computeCapProbs(P, tau)
    print("Final Min Cap Prob: ")
    print(tf.math.reduce_min(F))
    print("Final P: ")
    print(P)
    return P, F

# TESTING ----------------------------------------------------------------------------------
np.set_printoptions(linewidth=np.inf)
A = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
P0 = initRandP(A)
tau = 3
# print("P0 = ")
# print(P0)

# deltaP = 0.2*initRandP(np.ones([6, 6]))

# Ptest = P0 + deltaP
# print("Ptest = ")
# print(Ptest)

# # print("P0 = ")
# # print(P0)
# # print("deltaP = ")
# # print(deltaP)
# # print("Ptest = ")
# # print(Ptest)

# newP = projOntoSimplex(Ptest)
# print("newP = ")
# print(newP)

start_time = time.time()
[P, F] = gradAscentTF(P0, tau, 0.05, 1000)

print("--- Optimization took: %s seconds ---" % (time.time() - start_time))
print("P0 = ")
print(P0)
print("P_final = ")
print(P)

# P0 = tf.convert_to_tensor(P0)
# n = tf.shape(P0)[0]
# # print(n)

# print("P0 = ")
# print(P0)

# diag = tf.linalg.diag_part(P0, k=0, padding_value=0)
# print("diag = ")
# print(diag)

# Pdiag = tf.linalg.diag(tf.linalg.diag_part(P0, k=0, padding_value=0), k=0, num_rows=n, num_cols=n, padding_value=0)
# print("Pdiag = ")
# print(Pdiag)


# tf.linalg.diag_part(
#     input, name='diag_part', k=0, padding_value=0,
#     align='RIGHT_LEFT'
# )