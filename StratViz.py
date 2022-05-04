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

# Plot transition probabilities for each pair of nodes for the given P matrices,
# averaged across the results of each optimization for varying initial P
# Takes a list of initial and optimized P matrices
def plotOptTransProbs2D(optPMats, folderpath):
    n = optPMats[0].shape[0]
    probs = np.linspace(1, n**2, n**2)
    avgOptPmat = np.mean(optPMats, axis=0)
    stddevOptPmat = np.std(optPMats, axis=0)
    stddevpercOptPmat = 100*np.std(optPMats, axis=0)/np.mean(optPMats, axis=0)
    avgOptPvec = avgOptPmat.flatten('F')
    stddevOptPvec = stddevOptPmat.flatten('F')
    stddevpercOptPvec = stddevpercOptPmat.flatten('F')
    plt.figure()
    plt.scatter(probs, avgOptPvec, marker=".")
    plt.xlabel("Pvec Index")
    plt.ylabel("Avg Optimized Transition Probability")
    plt.title("Avg Optimized Transition Probabilies")
    plt.savefig(folderpath + "/avgOptP")
    plt.close()
    plt.figure()
    plt.scatter(probs, stddevOptPvec, marker=".")
    plt.xlabel("Pvec Index")
    plt.ylabel("Std Dev of Optimized Transition Probability")
    plt.title("Std Dev of Optimized Transition Probabilies")
    plt.savefig(folderpath + "/stddevOptP")
    plt.close()
    plt.figure()
    plt.scatter(probs, stddevpercOptPvec, marker=".")
    plt.xlabel("Pvec Index")
    plt.ylabel("(Std Dev/Avg) of Optimized Transition Probability (%)")
    plt.title("Std Dev of Optimized Transition Probabilies")
    plt.savefig(folderpath + "/stddevpercOptP")
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
def drawEnvGraph(A, graphNum, folderPath, save=True):
    G = genGraph(A)
    GViz = nx.nx_agraph.to_agraph(G)
    setEdgeLength(GViz, A, 2)
    GViz.graph_attr["nodesep"] = 0.5
    if(save):
        GViz.write(folderPath + "/A" + str(graphNum) +  ".dot")
        GViz.layout()
        GViz.draw(folderPath + "/A" + str(graphNum) +  ".png", prog="sfdp")
    return GViz

# Save the environment graph with edges labeled with transition probabilities:
def drawTransProbGraph(A, P, graphName, folderPath, Pnum=None, save=True):
    GViz = drawEnvGraph(A, graphName, folderPath, save=False)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                e = GViz.get_edge(i, j)
                e.attr["xlabel"] = "{:.3f}".format(P[i, j])
                e.attr["fontsize"] = 10.0
    if(save):
        GViz.write(folderPath + "/" + graphName + "_P" + str(Pnum) + ".dot")
        GViz.layout()
        GViz.draw(folderPath + "/" + graphName + "_P" + str(Pnum) +  ".png", prog="sfdp")
    return GViz

# Save the environment graph with edges labeled with optimized transition probabilities, 
# MCP listed, and node pair corresponding to MCP highlighted:
def drawOptGraphs(A, optPmats, graphName, optPnums, folderPath):
    for k in range(len(optPmats)):
        GViz = drawTransProbGraph(A, optPmats[k], graphName, folderPath, optPnums[k], save=False)
        GViz.layout()
        GViz.draw(folderPath + "/gviz" + "_opt_P_" + str(optPnums[k]) +  ".png", prog="sfdp")

# Save the environment graph with edges labeled with average optimized transition probabilities, 
# MCP listed, and node pair corresponding to MCP highlighted:
def drawAvgOptGraph(A, avgOptPmat, folderPath):
    GViz = drawTransProbGraph(A, avgOptPmat, None, folderPath, save=False)
    GViz.layout()
    GViz.draw(folderPath + "/gviz" + "_avg_opt_P" + ".png", prog="sfdp")

# Save the environment graph with edges labeled with average optimized transition probabilities, 
# edge line weights set proportional to trans probs, MCP listed, and node pair corresponding to 
# MCP highlighted:
def drawAvgOptGraphWeighted(A, avgOptPmat, folderPath):
    GViz = drawEnvGraph(A, None, folderPath, save=False)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                e = GViz.get_edge(i, j)
                e.attr["xlabel"] = "{:.3f}".format(avgOptPmat[i, j])
                e.attr["fontsize"] = 10.0
                e.attr["penwidth"] = str(10*avgOptPmat[i, j])
                gscale = int(255 - np.around(255*avgOptPmat[i, j]))
                hex_gscale = hex(gscale)
                hex_gscale = hex_gscale[2:]
                if len(hex_gscale) == 1:
                    hex_gscale = "0" + hex_gscale
                e.attr["color"] = "#" + hex_gscale + hex_gscale + hex_gscale
                if avgOptPmat[i, j] == 0:
                    GViz.delete_edge(i, j)
    GViz.layout()
    GViz.draw(folderPath + "/gviz" + "_avg_opt_P_linewt" + ".png", prog="sfdp")

# Parse graph name (test set subdirectory) to get graph number and tau.
# Graph name assumed to be in format: "graphX_tauY", returns X, Y
def parseGraphName(graphName):
    graphNum = graphName[(graphName.find("graph") + 5):graphName.find("_")]
    tau = graphName[(graphName.find("tau") + 3):len(graphName)]
    return graphNum, tau

# Parse metric filename to get run number and metric name.
# Filename assumed to be in format: "metricName_RunNum.csv"
def parseMetricName(metricName):
    mName = metricName[0:metricName.find("_")]
    runNum = metricName[(metricName.rfind("_") + 1):metricName.find(".csv")]
    return mName, runNum

# Search through "directory", collecting all files containing data for "metric".
# Return paths to metric data files, as well as corresponding iteration files.
def findOverlayFiles(directory, metric):
    iterFilePaths = {}
    metricFilePaths = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        tMetric, tRunNum = parseMetricName(filename)
        if os.path.isfile(f) and filename.find("iters") != -1:
            iterFilePaths[tRunNum] = f
        if os.path.isfile(f) and (filename.find("iters") == -1):
            if(metric == tMetric):
                metricFilePaths[tRunNum] = f
    return iterFilePaths, metricFilePaths


# Visualize metrics related to the optimization process:
def visMetrics(test_set_name, overlay=False):
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, "Results/test_set_" + test_set_name)
    for subdir in os.listdir(test_dir):
        if(subdir.find("graph") != -1):
            graphName = subdir
            graphNum, tau = parseGraphName(graphName)
            graph_dir = os.path.join(test_dir, graphName)
            metrics_dir = os.path.join(graph_dir, "metrics")
            met_vis_dir = os.path.join(graph_dir, "metrics_visualization")
            if not os.path.isdir(met_vis_dir):
                os.mkdir(met_vis_dir)
            usedFilePaths = []
            for filename in os.listdir(metrics_dir):
                i = os.path.join(metrics_dir, filename)
                if os.path.isfile(i):
                    if filename.find("iters") != -1:
                        _, runNum = parseMetricName(filename)
                        with open(i) as iters:
                            iters = np.loadtxt(iters, delimiter=',')
                        for filename in os.listdir(metrics_dir):
                            m = os.path.join(metrics_dir, filename)
                            if os.path.isfile(m) and (filename.find("iters") == -1) and filename not in usedFilePaths:
                                metricName, mRunNum = parseMetricName(filename)
                                if(mRunNum == runNum and metricName.find("final") == -1):
                                    usedFilePaths.append(filename)
                                    with open(m) as metric:
                                        metric = np.loadtxt(metric, delimiter = ',')
                                        plt.figure()
                                        plt.xlabel("Iteration Number")
                                        plt.ylabel(metricName)
                                        if not overlay:
                                            plt.plot(iters, metric, label="Run " + runNum)
                                            plt.title(graphName + " " + metricName + " Run " + runNum)
                                            plotpath = os.path.join(met_vis_dir, metricName + "_" + runNum)
                                        if overlay:
                                            plt.title(graphName + " " + metricName)
                                            plotpath = os.path.join(met_vis_dir, metricName)
                                            iterFiles, metricFiles = findOverlayFiles(metrics_dir, metricName)
                                            for r in iterFiles.keys():
                                                with open(iterFiles[r]) as tIters:
                                                    tIters = np.loadtxt(tIters, delimiter=',')
                                                with open(metricFiles[r]) as metric:
                                                    usedFilePaths.append(metricFiles[r])
                                                    metric = np.loadtxt(metric, delimiter = ',')
                                                plt.plot(tIters, metric, label="Run " + r)
                                        if metricName != "Pdiffs" and metricName != "AbsPdiffs":
                                            plt.legend(loc='lower right')
                                        plt.savefig(plotpath, bbox_inches = "tight")
                                        plt.close()



# Visualize initial and optimized P matrices: 
def visResults(test_set_name):
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, "Results/test_set_" + test_set_name)
    gcount = 0
    for subdir in os.listdir(test_dir):
        if(subdir.find("graph") != -1):
            graph_name = subdir
            graph_dir = os.path.join(test_dir, graph_name)
            res_vis_dir = os.path.join(graph_dir, "results_visualization")
            if not os.path.isdir(res_vis_dir):
                os.mkdir(res_vis_dir)
            initPMats = [] 
            optPMats = []
            initRunNums = []
            optRunNums = []
            for filename in os.listdir(graph_dir):
                runNum = filename[filename.rfind("_") + 1:filename.find(".csv")]
                f = os.path.join(graph_dir, filename)
                if(filename.find("A.csv") != -1):
                    with open(f) as A:
                        A = np.loadtxt(A, delimiter = ',')
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
            plotpath = os.path.join(res_vis_dir, "P_plot2D")
            plotTransProbs2D(initPMats, optPMats, initRunNums, optRunNums, graph_name, plotpath)
            plotOptTransProbs2D(optPMats, res_vis_dir)
            avgOptPMat = np.mean(optPMats, axis=0)
            drawAvgOptGraphWeighted(A, avgOptPMat, res_vis_dir)
            drawOptGraphs(A, optPMats, graph_name, optRunNums, res_vis_dir)
            gcount = gcount + 1
            print("Number of graphs processed: " + str(gcount)) 
            
def visMCPs(test_set_name):
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, "Results/test_set_" + test_set_name)
    graphNums = []
    taus = []
    mcpAvgs = []
    mcpStdDevs = []
    mcpStdDevPercents = []
    gcount = 0
    for subdir in os.listdir(test_dir):
        if(subdir.find("graph") != -1):
            graph_name = subdir
            graphNum, tau = parseGraphName(graph_name)
            graphNums.append(graphNum)
            taus.append(tau)
            graph_dir = os.path.join(test_dir, graph_name)
            info_filepath = os.path.join(graph_dir, "info.txt")
            with open(info_filepath) as info:
                line = True
                while line:
                    line = info.readline()
                    if(line.find("Final MCP achieved") != -1):
                        mcpString = line[line.find("[") + 1:line.find("]")]
                        mcps = mcpString.split()
                        mcps = list(map(float, mcps))
                        mcpAvgs.append(np.mean(mcps))
                        mcpStdDevs.append(np.std(mcps))
                        mcpStdDevPercents.append(100*(np.std(mcps)/np.mean(mcps)))
            gcount = gcount + 1
            print("Number of graphs processed: " + str(gcount)) 
    graphNums = list(map(int, graphNums))
    taus = list(map(int, taus))
    plt.figure()
    plt.scatter(graphNums, mcpAvgs)
    plt.xlabel("Graph Number")
    plt.ylabel("Average Optimized MCP")
    plt.savefig(test_dir + "/avgMCPs")
    plt.close()
    plt.figure()
    plt.scatter(graphNums, mcpStdDevs)
    plt.xlabel("Graph Number")
    plt.ylabel("Std Deviation of Optimized MCPs")
    plt.savefig(test_dir + "/stddevMCPs")
    plt.close()
    plt.figure()
    plt.scatter(graphNums, mcpStdDevPercents)
    plt.xlabel("Graph Number")
    plt.ylabel("(Std Deviation/Avg) of Optimized MCPs (%)")
    plt.savefig(test_dir + "/stddevpercMCPs")
    plt.close()


            
# TESTING -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    test_set_name = "2"
    # visMetrics(test_set_name, overlay=True)
    visResults(test_set_name)
    # visMCPs(test_set_name)

    # gscale = int(np.around(255*0.01))
    # hex_gscale = hex(gscale)
    # print(hex_gscale)

    # x = np.array([0, 1, 2, 3])
    # y = np.array([[0, 0, 0], [1, 2, 1], [2, 4, 1], [3, 6, 0]])
    # print(y.shape)

    # plt.figure()
    # plt.plot(x, y)
    # plt.savefig("testfig", bbox_inches = "tight")


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