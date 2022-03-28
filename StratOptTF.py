# Optimization of the performance of stochastic surveillance strategies using automatic differentiation
import time
import random
import numpy as np
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # FIX LATER
sys.path.insert(0, 'C:/Users/conno/anaconda3/envs/research_env/Lib/site-packages/')
import tensorflow as tf
from StratComp import *

# Compute first hitting time probability matrices (up to tau time steps) for fixed P matrix 
def computeFHTProbMatsTF(P, tau):  #DEAL WITH TF ITEM ASSIGNMENT ISSUE
    # print("tau = " + str(tau))
    n = tf.shape(P)[0]
    # F = tf.zeros([n, n, tau], dtype='float64')
    F = tf.reshape(P, (n, n, 1))
    F_last = P
    # F[:, :, 0] = P
    for i in range(1, tau):
        # F[:, :, i] = tf.linalg.matmul(P, (F[:, :, i - 1] - tf.linalg.diag(tf.linalg.diag_part(F[:, :, i - 1], k=0, padding_value=0), k=0, num_rows=n, num_cols=n, padding_value=0)))
        F_new = tf.linalg.matmul(P, (F_last - tf.linalg.diag(tf.linalg.diag_part(F_last, k=0, padding_value=0), k=0, num_rows=n, num_cols=n, padding_value=0)))
        F = tf.concat([F, tf.reshape(F_new, (n , n, 1))], 2)
        F_last = F_new
    return F

# Compute capture probabilities for each pair of nodes for fixed P matrix
def computeCapProbsTF(P, tau):
    FHTmats = computeFHTProbMatsTF(P, tau)
    capProbs = tf.math.reduce_sum(FHTmats, axis=2)
    return capProbs

# Project the given trans prob vector onto nearest point on probability simplex, if applicable
# See https://arxiv.org/abs/1309.1541 for explanation of the algorithm used here
def projOntoSimplexTF(P): 
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

def compCPJacTF(P, tau):
    with tf.GradientTape() as tape:
        tape.watch(P)
        F = computeCapProbsTF(P, tau)
    dP = tape.jacobian(F, P)
    return dP      

# TO DO: update and include zeroCPJacCols???
def gradAscentTF(P0, tau0, eps, iterations):
    P = tf.Variable(P0)
    tau = tf.constant(tau0)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            # tau = tf.stop_gradient(tau)
            tape.watch(P)
            F = computeCapProbsTF(P, tau)
            mcp = tf.math.reduce_min(F)
        dP = tape.gradient(mcp, P)      # WHAT DOES THIS DO WITH NONUNIQUE MINIMA?
        P = P + eps*dP
        P = projOntoSimplexTF(P)
        if i % 100 == 0:
            print("Minimum Capture Probability at iteration " + str(i) + ":")
            print(mcp)
    F = computeCapProbsTF(P, tau)
    print("Final Min Cap Prob: ")
    print(tf.math.reduce_min(F))
    print("Final P: ")
    print(P)
    return P, F

# TESTING ----------------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    # np.set_printoptions(suppress=True)
    A = genStarG(4)
    P0 = initRandP(A)
    tau = 3
    # print("P0 = ")
    # print(P0)

    # start_time = time.time()
    # [P, F] = gradAscentTF(P0, tau, 0.05, 1000)

    # print("--- Optimization took: %s seconds ---" % (time.time() - start_time))
    # print("P0 = ")
    # print(P0)
    # print("P_final = ")
    # print(P)

    P = tf.Variable(P0)
    tau = tf.constant(tau)

    start_time = time.time()
    J = compCPJacTF(P, tau)
    print("--- Jacobian Computation took: %s seconds ---" % (time.time() - start_time))
    print("Computed Jacobian = ")
    print(J)

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