# Visualization of the performance of stochastic surveillance strategies
import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
from StratComp import *

# Plot transition probabilities for each pair of nodes for the given P matrices
def plotTransProbs2D(PMats, title):
    n = PMats.shape[0]
    probs = np.linspace(1, 36, 36)
    fig = plt.figure()
    for k in range(PMats.shape[2]):
        Pvec = PMats[:, : , k].flatten('F')
        plt.scatter(probs, Pvec)
    plt.xlabel("Pvec Index")
    plt.ylabel("Probability")
    plt.title(title)
    plt.show()
                
# Plot capture probabilities for each pair of nodes for fixed P matrix
def plotCapProbs3D(capProbs):
    roboLocs = np.arange(capProbs.shape[0])
    intruLocs = np.arange(capProbs.shape[1])
    X, Y = np.meshgrid(roboLocs, intruLocs)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, capProbs)
    ax.set_xlabel('Robot Location')
    ax.set_ylabel('Intruder Location')
    ax.set_zlabel('Capture Probability')
    plt.show()

# Plot the capture probabilities as a function of P_ij
def compCPVarP(P, tau, i, j):
    capProbs = np.full([ P.shape[0], P.shape[1], 51], np.NaN)
    for k in range(51):
        # Set the (i, j) element of transition probability matrix:
        P[i, j] = k/50
        # Normalize all the other entries in row i to generate valid prob dist:
        remRowSum = np.sum(P[i, :]) - P[i, j]
        for col in range(P.shape[1]):
            if col != j:
                P[i, col] = P[i, col]*(1 - P[i, j])/remRowSum
        capProbs[:, :, k] = computeCapProbs(P, tau)
    pRange = np.linspace(0, 1, 51)
    for row in range(P.shape[0]):
        for col in range(P.shape[1]):
            plt.plot(pRange, capProbs[row, col, :], label = "(" + str(row) + ", " + str(col) + ")")
    plt.xlabel("P(" + str(i) + ", " + str(j) + ") value")
    plt.ylabel("Capture Probabilities")
    plt.legend(bbox_to_anchor=(1.05, 0.95), loc='upper right', borderaxespad=-2)
    plt.title("Capture Probabilities for tau = " + str(tau) + " and varying P(" + str(i) + ", " + str(j) + ")")
    plt.show()
    return capProbs

# Generate a NetworkX graph from adjacency matrix A
def genGraph(A):
    # A = A - np.diag(np.diag(A))
    temp = nx.DiGraph()
    G = nx.from_numpy_matrix(A, create_using=temp)
    return G

# Set the length of all edges in the graph
def setEdgeLength(G, A, len):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                edge = G.get_edge(i, j)
                edge.attr["len"] = len

# Save the environment graph without edge labels:
def drawEnvGraph(A, graphNum, folderPath):
    G = genGraph(A)
    GViz = nx.nx_agraph.to_agraph(G)
    setEdgeLength(GViz, A, 2)
    GViz.graph_attr["nodesep"] = 0.5
    GViz.graph_attr["len"] = 4
    GViz.write(folderPath + "/A" + str(graphNum) +  ".dot")
    GViz.layout()
    GViz.draw(folderPath + "/A" + str(graphNum) +  ".png", prog="sfdp")


# TESTING -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # initialize transition prob matrix
    # # P = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # initialize transition prob matrix
    # P = np.matmul(np.diag(1/np.sum(P, axis=1)), P)   # normalize to generate valid prob dist

    tau = 2  # attack duration

    # plotCapProbs3D(capProbs)

    # capProbs = compCPVarP(P, 3, 0, 0)

    # A = genStarG(9)
    A = np.array([[1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 1]])
    # print(A)
    # P0 = np.array([[0, 0.2, 1 - 0.2], [1, 0, 0], [1, 0, 0]])
    P0 = initRandP(A)



    # # fig2 = plt.figure()
    # # ax2 = plt.plot()
    # G = genGraph(A)
    # # nx.draw(G, with_labels=True, font_weight='bold')
    # # plt.show()

    # GViz = nx.nx_agraph.to_agraph(G)

    # center = GViz.get_node(0)
    # center.attr["color"] = "blue"
    # # center.attr["fixedsize"] = True
    # # center.attr["imagescale"] = True
    # # center.attr["image"] = "small robot.png"

    # # intruLoc = GViz.get_node(3)
    # # intruLoc.attr["color"] = "red"
    # # intruLoc.attr["fixedsize"] = True
    # # intruLoc.attr["image"] = "small intruder.png"

    # for i in range(A.shape[0]):
    #     for j in range(A.shape[1]):
    #         if A[i, j] == 1:
    #             e = GViz.get_edge(i, j)
    #             e.attr["xlabel"] = "{:.3f}".format(P0[i, j])
    #             # e.attr["decorate"] = True
    #             # e.attr["labelfloat"] = True
    #             e.attr["fontsize"] = 10.0

    # # e = GViz.get_edge(center, intruLoc)
    # # e.attr["label"] = "P_03"
    # # e.attr["color"] = "blue"

    # setEdgeLength(GViz, A, 2)

    # GViz.graph_attr["nodesep"] = 0.5
    # GViz.graph_attr["len"] = 4
    # GViz.write("test_graph.dot")
    # GViz.layout()
    # GViz.draw("test_graph.png", prog="sfdp")
    # # GViz.draw("test_graph.png", prog="neato")

