# Just a script to use for testing other components of the robotic surveillance code 
import os
import time

import numpy as np
import jax.numpy as jnp

import StratCompJax as scj
import StratOptOptax as soo
import StratViz as sv


def NUDEL_FHT_TEST():
    # Testing NUDEL FHT prob mat computation in StratCompJax:
    n = 6
    w_LB = 1
    w_UB= 1
    tau = 4
    A, W, w_max, g_name = scj.gen_rand_NUDEL_star_G(n, w_UB, w_LB)

    # width_vec = np.array([1, 1])
    # height_vec = np.array([1])
    # A, W, g_name = scj.gen_NUDEL_grid_G(width_vec, height_vec)
    sv.draw_env_graph(W, g_name, os.getcwd(), show_edge_lens=True)

    # w_max = int(jnp.max(W))

    print("W = ")
    print(W)

    P = scj.init_rand_P(A)
    print("P = ")
    print(P)
    n = jnp.shape(P)[0]

    sv.draw_trans_prob_graph(A, P, g_name, os.getcwd(), P_num=0)

    F_v1 = scj.compute_NUDEL_FHT_probs_vec(P, W, w_max, tau)
    P_vec = jnp.reshape(P, (n**2), order="F")
    F_v2 = scj.compute_FHT_probs_vec(P_vec, tau)

    print("F_v1 = ")
    print(F_v1)
    print("F_v2 = ")
    print(F_v2)

    F_v1 = np.array(F_v1)
    F_v2 = np.array(F_v2)

    # for i in range(20):
    #     A, W, g_name = scj.gen_rand_NUDEL_star_G(n, w_UB)
    #     w_max = int(jnp.max(W))
    #     P = scj.init_rand_P(A)
    #     start_time = time.time()
    #     F_v = scj.compute_NUDEL_FHT_probs_vec(P, W, w_max, tau)
    #     print("t_" + str(i + 1) + " = " + str(time.time() - start_time) + " seconds")

    print("F_v abs diff sum = ")
    print(np.sum(np.abs(F_v1 - F_v2)))



if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)

    # NUDEL_FHT_TEST()

    A1, g_name1 = scj.gen_grid_G(5, 5)
    print("Unholy grid = ")
    print(A1)
    sv.draw_env_graph(A1, g_name1, os.getcwd())

    m_nodes = [np.array([2, 1]), np.array([2, 3])]
    A2, g_name2 = scj.gen_holy_grid_G(5, 5, m_nodes)
    print("Holy grid = ")
    print(A2)

    sv.draw_env_graph(A2, g_name2, os.getcwd())
    





