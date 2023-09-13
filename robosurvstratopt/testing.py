# Just a script to use for testing other components of the robotic surveillance code 
import csv
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy

import graph_gen as gg
import graph_comp as gc
import strat_comp as scj
import strat_opt as soo
import strat_viz as sv

def NUDEL_FHT_TESTS(test_grid=True, test_star=True, test_JIT=True):
    # Testing NUDEL FHT prob mat computation in StratCompJax:
    if test_star:
        n = 6
        tau = 6
        w_LB = 1
        w_UB= 2
        A, W, w_max, g_name = gg.gen_rand_NUDEL_star_G(n, w_UB, w_LB)
        sv.draw_env_graph(W, g_name, os.getcwd(), show_edge_lens=True)
        print("W = ")
        print(W)
        P = scj.init_rand_P(A)
        print("P = ")
        print(P)
        n = jnp.shape(P)[0]
        sv.draw_trans_prob_graph(A, P, g_name, os.getcwd(), P_num=0)
        F_v1 = scj.compute_NUDEL_FHT_probs_vec(P, W, w_max, tau)
        print("F_v1 = ")
        print(F_v1)
        A, W, w_max, g_name = gg.gen_rand_NUDEL_star_G(n, w_UB)
        old_w_max = 0
        for i in range(20):
            if w_max == old_w_max:
                old_w_max = w_max
                print("Retracing for new w_max value...")
            P = scj.init_rand_P(A)
            start_time = time.time()
            F_v = scj.compute_NUDEL_FHT_probs_vec(P, W, w_max, tau)
            print("t_" + str(i + 1) + " = " + str(time.time() - start_time) + " seconds")
            A, W, w_max, g_name = gg.gen_rand_NUDEL_star_G(n, w_UB)
    if test_grid:
        n = 6
        width_vec = np.array([2, 3, 2])
        height_vec = np.array([1, 2])
        A, W, g_name = gg.gen_NUDEL_grid_G(width_vec, height_vec)
        tau = gg.graph_diam(A)
        sv.draw_env_graph(W, g_name, os.getcwd(), show_edge_lens=True)
        print("W = ")
        print(W)
        P = scj.init_rand_P(A)
        print("P = ")
        print(P)
        n = jnp.shape(P)[0]
        sv.draw_trans_prob_graph(A, P, g_name, os.getcwd(), P_num=0)
        F_v1 = scj.compute_NUDEL_FHT_probs_vec(P, W, w_max, tau)
        print("F_v1 = ")
        print(F_v1)

def HOLY_GRID_TESTS():
    # Testing generator function for grid graph with holes:
    A1, g_name1 = gg.gen_grid_G(5, 5)
    print("Unholy grid = ")
    print(A1)
    sv.draw_env_graph(A1, g_name1, os.getcwd())
    m_nodes = [np.array([2, 1]), np.array([2, 3])]
    A2, g_name2 = gg.gen_holy_grid_G(5, 5, m_nodes)
    print("Holy grid = ")
    print(A2)

    sv.draw_env_graph(A2, g_name2, os.getcwd())

def F0_REQ_TESTS():
    # Trying to remember why I'm passing F0 as a parameter in FHT probs computation...
    # Think it was for JIT reasons? Want to verify if it's actually necessary and
    # do a speed comparison with and without passing it. 
    A, g_name = gg.gen_grid_G(50, 20)
    n = jnp.shape(A)[0]
    tau = 5
    F0 = jnp.full((n, n, tau), np.NaN)
    num_iters = 100
    num_tests = 20
    init_Ps = scj.init_rand_Ps(A, num_iters)
    t_time = jnp.zeros(num_tests)
    for test_num in range(num_tests):
        # with F0 passed as param:
        series_start_time = time.time()
        for i in range(num_iters):
            P = init_Ps[:, :, i]
            start_time = time.time()
            F_v = scj.compute_cap_probs(P, F0, tau)
            i_time = time.time() - start_time
            # print("t_" + str(i + 1) + " = " + str(i_time) + " seconds")
            t_time = t_time.at[test_num].set(t_time[test_num] + i_time)
        print("Time elapsed for " + str(num_iters) + " iterations with F0 passing = " + str(time.time() - series_start_time) + " seconds")
    print("Iter time sums = " + str(t_time) + " seconds")
    print("Avg post-trace iter time sum = " + str(jnp.mean(t_time[:num_tests - 1])) + " seconds")
    t_time = jnp.zeros(num_tests)
    for test_num in range(num_tests):
        # without F0 passed as param:
        series_start_time = time.time()
        for i in range(num_iters):
            P = init_Ps[:, :, i]
            start_time = time.time()
            F_v = scj.compute_cap_probs_NF0(P, tau)
            i_time = time.time() - start_time
            # print("t_" + str(i + 1) + " = " + str(i_time) + " seconds")
            t_time = t_time.at[test_num ].set(t_time[test_num] + i_time)
        print("Time elapsed for " + str(num_iters) + " iterations without F0 passing = " + str(time.time() - series_start_time) + " seconds")
    print("Iter time sums = " + str(t_time) + " seconds")
    print("Avg post-trace iter time sum = " + str(jnp.mean(t_time[:num_tests - 1])) + " seconds")
    
def CSV_RW_TEST():
    # writing multiple P matrices to same CSV
    num_Ps = 5
    n = 4
    A, g_name = gg.gen_star_G(n)
    rand_Ps = scj.init_rand_Ps(A, num_Ps)
    start_write_time = time.time()
    with open(os.getcwd() + '/P_mats.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["num_Ps = " + str(num_Ps), "n = " + str(n)])
        for P_num in range(num_Ps):
            writer.writerows(rand_Ps[:, :, P_num])     
    writing_time = time.time() - start_write_time
    print("Writing P matrices to CSV took " + str(writing_time) + " seconds")
    # reading from CSV and loading each P mat to a jnp devicearray
    req_P_nums = [1, 3]
    start_read_time = time.time()
    with open(os.getcwd() + '/P_mats.csv', 'r') as csv_rfile:
        first_line = next(csv_rfile)
        r_num_Ps = int(first_line[0].split.pop())
        r_nsq = int(first_line[1].split.pop()) 
        req_P_firstrows = r_nsq*(req_P_nums - 1)
        req_P_lastrows = req_P_firstrows + r_nsq
        reader = csv.reader(csv_rfile)
        req_P_mats = jnp.full([r_nsq, r_nsq, req_P_nums], np.NaN)
        load_row = False
        for line_num, line in enumerate(reader):
            if line_num in req_P_firstrows:
                req_P_num = req_P_firstrows.index()
                load_row = True
                P_tmp = np.full([r_nsq, r_nsq])
            if load_row:
                row = np.fromstring(line, dtype=float, sep=',')

# Extract and print out capture probabilities for all leaf node pairs:
def GET_LEAF_CP_TEST(bin_adj_path, opt_P_path):
    with open(bin_adj_path) as bin_adj_file:
        A = np.loadtxt(bin_adj_file, delimiter=',')
    with open(opt_P_path) as opt_P_file:
        opt_P = np.loadtxt(opt_P_file, delimiter=',')
    opt_P = jnp.array(opt_P)
    tau = 7
    F = scj.compute_cap_probs_NF0(opt_P, tau)
    leaf_node_pairs = gg.get_leaf_node_pairs(A)
    diam_pairs = gg.get_diametric_pairs(A)
    dp_cap_probs = scj.compute_diam_pair_cap_probs(F, diam_pairs)
    print("Diametric pairs cap probs:")
    print(dp_cap_probs)
    dp_CP_var = scj.compute_diam_pair_CP_variance(F, diam_pairs)
    print("Diametric pairs cap probs variance:")
    print(dp_CP_var)
    return A, leaf_node_pairs

# Examine shortest-path-distances, shortest-path-cap-probs, and tau-cap-probs
#  for some optimized strategies for tree graphs:
def COMPARE_SPD_SPCP_TAUCP():
    opt_P_path = os.getcwd() + "/Results/test_set_InitP100_Study_D3RandTree1/test13_randomtree_D3_N12_tau5/results/opt_P_87.csv"
    bin_adj_path = os.getcwd() + "/Results/test_set_InitP100_Study_D3RandTree1/test13_randomtree_D3_N12_tau5/A.csv"
    # opt_P_path = os.getcwd() + "/Results/test_set_InitP100_Study_D3RandTree1/test18_randomtree_D3_N12_tau8/results/opt_P_14.csv"
    # bin_adj_path = os.getcwd() + "/Results/test_set_InitP100_Study_D3RandTree1/test18_randomtree_D3_N12_tau8/A.csv"
    # opt_P_path = os.getcwd() + "/Results/test_set_InitP100_Study_D3RandTree1/test16_randomtree_D3_N12_tau5/results/opt_P_23.csv"
    # bin_adj_path = os.getcwd() + "/Results/test_set_InitP100_Study_D3RandTree1/test16_randomtree_D3_N12_tau5/A.csv"
    with open(bin_adj_path) as bin_adj_file:
        A = np.loadtxt(bin_adj_file, delimiter=',')
    with open(opt_P_path) as opt_P_file:
        opt_P = np.loadtxt(opt_P_file, delimiter=',')
    # identify leaf node pairs and their shortest-path distances for given acyclic graph:
    leaf_node_pairs = gg.get_leaf_node_pairs(A)
    spds = gg.get_shortest_path_distances(A, leaf_node_pairs)
    # get tau-capture probabilities for each leaf node pair in acyclic graph:
    tau = 5
    FHT = scj.compute_FHT_probs_NF0(opt_P, tau)
    leaf_tau_CPs = jnp.zeros((jnp.shape(leaf_node_pairs)[0], 3))
    leaf_tau_CPs = leaf_tau_CPs.at[:, :2].set(leaf_node_pairs)
    F = scj.compute_cap_probs_NF0(opt_P, tau)
    for pair_ind in range(jnp.shape(leaf_node_pairs)[0]):
        leaf_tau_CPs = leaf_tau_CPs.at[pair_ind, 2].set(F[leaf_node_pairs[pair_ind, 0], leaf_node_pairs[pair_ind, 1]])
    # compute shortest-path capture probabilities (SPCPs) for each leaf node pair in acyclic graph:
    leaf_SPCPs = scj.compute_SPCPs(A, FHT, leaf_node_pairs)
    leaf_SPD_SPCP_tauCP = jnp.zeros((jnp.shape(spds)[0], 5))
    leaf_SPD_SPCP_tauCP = leaf_SPD_SPCP_tauCP.at[:, :3].set(spds)
    leaf_SPD_SPCP_tauCP = leaf_SPD_SPCP_tauCP.at[:, 3].set(leaf_SPCPs[:, 2])
    leaf_SPD_SPCP_tauCP = leaf_SPD_SPCP_tauCP.at[:, 4].set(leaf_tau_CPs[:, 2])
    # compute difference between SPCPs for i -> j vs j -> i, for i, j any leaf node pair (also tau-CP differences):
    sym_SPCP_diffs = jnp.zeros((jnp.shape(leaf_SPD_SPCP_tauCP)[0], 3))
    sym_SPCP_diffs = sym_SPCP_diffs.at[:, :2].set(leaf_node_pairs)
    sym_tauCP_diffs = jnp.zeros((jnp.shape(leaf_SPD_SPCP_tauCP)[0], 3))
    sym_tauCP_diffs = sym_tauCP_diffs.at[:, :2].set(leaf_node_pairs)
    for pair_ind in range(jnp.shape(leaf_SPD_SPCP_tauCP)[0]):
        robo_loc = leaf_SPD_SPCP_tauCP[pair_ind, 0]
        intru_loc = leaf_SPD_SPCP_tauCP[pair_ind, 1]
        for rev_pair_ind in range(jnp.shape(leaf_SPD_SPCP_tauCP)[0]):
            if leaf_SPD_SPCP_tauCP[rev_pair_ind, 0] == intru_loc and leaf_SPD_SPCP_tauCP[rev_pair_ind, 1] == robo_loc:
                sym_SPCP_diffs = sym_SPCP_diffs.at[pair_ind, 2].set(leaf_SPD_SPCP_tauCP[pair_ind, 3] - leaf_SPD_SPCP_tauCP[rev_pair_ind, 3])
                sym_tauCP_diffs = sym_tauCP_diffs.at[pair_ind, 2].set(leaf_SPD_SPCP_tauCP[pair_ind, 4] - leaf_SPD_SPCP_tauCP[rev_pair_ind, 4])
    # plot SPCPs and tau-CPs vs shortest-path-distance for each leaf node pair:
    plt.figure()
    plt.scatter(leaf_SPD_SPCP_tauCP[:, 2], leaf_SPD_SPCP_tauCP[:, 3], Color='g', label="Shortest-Path CP", s=0.4)
    plt.scatter(leaf_SPD_SPCP_tauCP[:, 2], leaf_SPD_SPCP_tauCP[:, 4], Color='r', label="tau-CP", s=0.4)
    plt.xlabel("Shortest-Path Distance")
    plt.ylabel("Probability Difference")
    plt.title("Leaf Node Pairs for Tree Graph")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/LeafNodeAnalysisTree3")
    plt.close()
    # plot difference between SPCP's for i -> j vs j -> i, for i, j any leaf node pair:
    plt.figure()
    plt.scatter(leaf_SPD_SPCP_tauCP[:, 2], sym_SPCP_diffs[:, 2], Color='g', label="Sym SPCP Diff", s=0.4)
    # plt.scatter(leaf_SPD_SPCP_tauCP[:, 2], sym_tauCP_diffs[:, 2], Color='r', label="Sym tau-CP Diff", s=0.4)
    plt.xlabel("Shortest-Path Distance")
    plt.ylabel("Probability Difference")
    plt.title("Leaf Node Pairs for Tree Graph")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/LeafNodeSymmetryAnalysisTree3")
    plt.close()
    # plot SPCPs and abs-val of difference between SPCP's for i -> j vs j -> i, for i, j any leaf node pair:
    plt.figure()
    plt.scatter(leaf_SPD_SPCP_tauCP[:, 2], leaf_SPD_SPCP_tauCP[:, 3], Color='g', label="Shortest-Path CP", s=0.5)
    plt.scatter(leaf_SPD_SPCP_tauCP[:, 2], jnp.abs(sym_SPCP_diffs[:, 2]), Color='r', label="Sym SPCP Diff", s=0.5)
    # plt.scatter(leaf_SPD_SPCP_tauCP[:, 2], sym_tauCP_diffs[:, 2], Color='r', label="Sym tau-CP Diff", s=0.4)
    plt.xlabel("Shortest-Path Distance")
    plt.ylabel("Probability Difference")
    plt.title("Leaf Node Pairs for Tree Graph")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/LeafNodeSymmetryDiffvsSPCPComparisonTree3")
    plt.close()

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    n = 3
    N = 2
    A = jnp.ones((n, n))
    # A = A.at[0, 1].set(0)
    # A = A.at[1, 0].set(0)
    As = jnp.ones((N, n, n))
    # pi = (0.4, 0.2, 0.25, 0.15)
    tau = 1
    # B = 108
    pi = (0.3, 0.3, 0.4)
    alpha = 1000
    # Ps = scj.init_rand_Ps(A, 2, 0)
    # Q = jnp.reshape(P, (n, n))
    # Q = (1/3)*jnp.ones((n, n))
    # print(Q)
    # w, vl, _ = scipy.linalg.eig(Q, left=True)
    # pi = ( vl[:,0] / jnp.sum(vl[:,0]) ).T
    # pi = tuple(float(jnp.real(x)) for x in pi)
    # print(pi)
    # print(jnp.dot(pi, Q))
    # P = (1/n)*jnp.ones((n, n))
    # Q = Q.at[0, 0].set(0)
    W = jnp.ones((n, n))
    # W = jnp.array([[1, 0, 2], [0, 1, 1], [2, 1, 1]])
    # print(W)
    # W = jnp.array([[1, 3, 3, 5, 4, 6, 3, 5, 7, 4, 6, 6],
    #         [3, 1, 5, 4, 2, 4, 4, 5, 5, 3, 5, 5],
    #         [3, 5, 1, 7, 6, 8, 3, 4, 9, 4, 8, 7],
    #         [6, 4, 7, 1, 5, 6, 4, 7, 5, 6, 6, 7],
    #         [4, 3, 6, 5, 1, 3, 5, 5, 6, 3, 4, 4],
    #         [6, 4, 8, 5, 3, 1, 6, 7, 3, 6, 2, 3],
    #         [2, 5, 3, 5, 6, 7, 1, 5, 7, 5, 7, 8],
    #         [3, 5, 2, 7, 6, 7, 3, 1, 9, 3, 7, 5],
    #         [8, 6, 9, 4, 6, 4, 6, 9, 1, 8, 5, 7],
    #         [4, 3, 4, 6, 3, 5, 5, 3, 7, 1, 5, 3],
    #         [6, 4, 8, 6, 4, 2, 6, 6, 4, 5, 1, 3],
    #         [6, 4, 6, 6, 3, 3, 6, 4, 5, 3, 2, 1]])
    w_max = int(jnp.max(W))
    init_Ps = scj.multi_init_rand_Ps(As, N, 1)
    P = init_Ps[0, :, :, :]
    print(P[0])
    print(P[1])
    D_idx = scj.precompute_weighted_Stackelberg(W, w_max, tau)
    combs, combs_len = scj.precompute_multi(n, N)
    # print(scj.loss_weighted_multi_LCP_pi(P, As, D_idx, combs, N, combs_len, W, w_max, tau, pi, alpha))
    print(scj.comp_avg_weighted_multi_LCP_pi_grad(P, As, D_idx, combs, N, combs_len, W, w_max, tau, pi, alpha))
    print(gc.gen_graph_code(jnp.ones((4, 4))))
    # w, vl, _ = scipy.linalg.eig(P[0], left=True)
    # pi_0 = ( vl[:,0] / jnp.sum(vl[:,0]) ).T
    # print(pi_0)
    # w, vl, _ = scipy.linalg.eig(P[1], left=True)
    # pi_1 = ( vl[:,0] / jnp.sum(vl[:,0]) ).T
    # print(pi_1)
    # eta = 0.5
    # N_eta = int(jnp.ceil(w_max/(eta*jnp.min(jnp.array(pi)))) - 1)
    # print(N_eta)
    # print(scj.compute_weighted_multi_cap_probs(P0, D_idx, combs, N, combs_len, W, w_max, tau))
    # print(scj.compute_weighted_multi_LCPs(P0, D_idx, combs, N, combs_len, W, w_max, tau))
    # print(scj.loss_weighted_multi_LCP(P0, As, D_idx, combs, N, combs_len, W, w_max, tau))
    # print(scj.comp_avg_weighted_multi_LCP_grad(P0, As, D_idx, combs, N, combs_len, W, w_max, tau))

    # combs = scj.precompute_weighted_multi_cap_probs(n, N)
    # print(combs)
    # combs_len = len(combs)
    # print(combs_len)
    # print(scj.compute_weighted_multi_cap_probs(Ps, D_idx, combs, N, combs_len, W, w_max, tau))
    # print(scj.comp_avg_weighted_multi_LCP_grad(Ps, As, D_idx, combs, N, combs_len, W, w_max, tau))
