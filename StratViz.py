# Visualization of the performance of stochastic surveillance strategies
import os
import numpy as np
import math
import csv
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import networkx as nx
import pygraphviz as pgv
from StratComp import *

# Plot transition probabilities for each pair of nodes for the given P matrices
# Takes a list of initial and optimized P matrices
def plotTransProbs2D(initPMats, optPMats, initRunNums, optRunNums, title, path):
    n = initPMats[0].shape[0]
    probs = np.linspace(1, n**2, n**2)
    plt.figure()
    for k in range(len(initPMats)):
        initPvec = initPMats[k].flatten('F')
        optPvec = optPMats[k].flatten('F')
        initNum = initRunNums[k]
        optNum = optRunNums[k]
        plt.scatter(probs, initPvec, marker=".", color="C" + str(initNum), label="Run " + initNum)
        plt.scatter(probs, optPvec, marker="*", color="C" + str(optNum))
    plt.xlabel("Pvec Index")
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend(loc='center right')
    plt.savefig(path)
    plt.close()


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

# Save the environment graph with edges labeled with transition probabilities:
def drawTransProbGraph(A, P, graphNum, folderPath):
    G = genGraph(A)
    GViz = nx.nx_agraph.to_agraph(G)
    setEdgeLength(GViz, A, 2)
    GViz.graph_attr["nodesep"] = 0.5
    GViz.graph_attr["len"] = 4
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                e = GViz.get_edge(i, j)
                e.attr["xlabel"] = "{:.3f}".format(P[i, j])
                # e.attr["decorate"] = True
                # e.attr["labelfloat"] = True
                e.attr["fontsize"] = 10.0
    GViz.write(folderPath + "/A" + str(graphNum) +  ".dot")
    GViz.layout()
    GViz.draw(folderPath + "/A" + str(graphNum) +  ".png", prog="sfdp")

# Visualize metrics related to the optimization process:
def visMetrics(test_set_name, overlay=False):
    cwd = os.getcwd()
    met_vis_dir = os.path.join(cwd, "Results/test_set_" + test_set_name + "/metrics_visualization")
    if not os.path.isdir(met_vis_dir):
        os.mkdir(met_vis_dir)
    metrics_dir = os.path.join(cwd, "Results/test_set_" + test_set_name + "/metrics")
    usedFilePaths = []
    for filename in os.listdir(metrics_dir):
        i = os.path.join(metrics_dir, filename)
        if os.path.isfile(i):
            graphNum, tau, runNum = parseGraphName(filename)
            if filename.find("iters") != -1:
                with open(i) as iters:
                    iters = np.loadtxt(iters, delimiter=',')
                for filename in os.listdir(metrics_dir):
                    m = os.path.join(metrics_dir, filename)
                    if os.path.isfile(m) and (filename.find("iters") == -1) and filename not in usedFilePaths:
                        mGraphNum, mTau, mRunNum = parseGraphName(filename)
                        metricName = filename[(filename.find("_", filename.find("tau")) + 1):filename.rfind("_")]
                        if(mGraphNum == graphNum and mTau == tau and mRunNum == runNum and metricName.find("final") == -1):
                            usedFilePaths.append(filename)
                            with open(m) as metric:
                                metric = np.loadtxt(metric, delimiter = ',')
                                plt.figure()
                                plt.xlabel("Iteration Number")
                                plt.ylabel(metricName)
                                plot_dir = os.path.join(met_vis_dir, "graph" + graphNum + "_tau" + tau)
                                if not os.path.isdir(plot_dir):
                                    os.mkdir(plot_dir)
                                if not overlay:
                                    plt.plot(iters, metric, label="Run " + runNum)
                                    plt.title("Graph " + graphNum + " " + metricName + " Run " + runNum)
                                    plotpath = os.path.join(plot_dir, metricName + "_" + runNum)
                                if overlay:
                                    plt.title("Graph " + graphNum + " " + metricName)
                                    plotpath = os.path.join(plot_dir, metricName)
                                    iterFiles, metricFiles = findOverlayFiles(metrics_dir, graphNum, tau, metricName)
                                    for r in iterFiles.keys():
                                        with open(iterFiles[r]) as tIters:
                                            tIters = np.loadtxt(tIters, delimiter=',')
                                        with open(metricFiles[r]) as metric:
                                            usedFilePaths.append(metricFiles[r])
                                            metric = np.loadtxt(metric, delimiter = ',')
                                        plt.plot(tIters, metric, label="Run " + r)
                                plt.legend(loc='lower right')
                                plt.savefig(plotpath, bbox_inches = "tight")
                                plt.close()

def parseGraphName(graphName):
    gName = graphName[0:graphName.find(".csv")]
    graphNum = gName[(gName.find("graph") + 5):gName.find("_")]
    tau = gName[(gName.find("tau") + 3):gName.find("_", gName.find("tau"))]
    runNum = gName[(gName.rfind("_") + 1):len(gName)]
    return graphNum, tau, runNum

def findOverlayFiles(directory, graphNum, tau, metric):
    iterFilePaths = {}
    metricFilePaths = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        tGraphNum, tTau, tRunNum = parseGraphName(filename)
        if os.path.isfile(f) and filename.find("iters") != -1:
            if(graphNum == tGraphNum and tau == tTau):
                iterFilePaths[tRunNum] = f
        if os.path.isfile(f) and (filename.find("iters") == -1):
            tMetric = filename[(filename.find("_", filename.find("tau")) + 1):filename.rfind("_")]
            if(graphNum == tGraphNum and tau == tTau and metric == tMetric):
                metricFilePaths[tRunNum] = f
    return iterFilePaths, metricFilePaths


# Visualize initial and optimized P matrices: 
def visResults(test_set_name):
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, "Results/test_set_" + test_set_name)
    res_vis_dir = os.path.join(test_dir, "results_visualization")
    if not os.path.isdir(res_vis_dir):
        os.mkdir(res_vis_dir)
    for subdir in os.listdir(test_dir):
        if(subdir.find("graph") != -1):
            graph_name = subdir
            graph_dir = os.path.join(test_dir, subdir)
            initPMats = [] 
            optPMats = []
            initRunNums = []
            optRunNums = []
            for filename in os.listdir(graph_dir):
                runNum = filename[filename.rfind("_") + 1:filename.find(".csv")]
                f = os.path.join(graph_dir, filename)
                if(filename.find("init_P") != -1):
                    with open(f) as initP:
                        initP = np.loadtxt(initP, delimiter = ',')
                        initPMats.append(initP)
                        initRunNums.append(runNum)
                if(filename.find("opt_P") != -1):
                    with open(f) as optP:
                        optP = np.loadtxt(optP, delimiter = ',')
                        optPMats.append(optP)
                        optRunNums.append(runNum)
            plotpath = os.path.join(res_vis_dir, graph_name)
            plotTransProbs2D(initPMats, optPMats, initRunNums, optRunNums, graph_name, plotpath)

            
# TESTING -------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    visMetrics("GridGraphTestingLongRunMCP", overlay=True)
    visResults("GridGraphTestingLongRunMCP")


    # P = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # initialize transition prob matrix
    # # P = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # initialize transition prob matrix
    # P = np.matmul(np.diag(1/np.sum(P, axis=1)), P)   # normalize to generate valid prob dist

    # tau = 2  # attack duration

    # plotCapProbs3D(capProbs)

    # capProbs = compCPVarP(P, 3, 0, 0)

    # # A = genStarG(9)
    # A = np.array([[1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 1]])
    # # print(A)
    # # P0 = np.array([[0, 0.2, 1 - 0.2], [1, 0, 0], [1, 0, 0]])
    # P0 = initRandP(A)

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