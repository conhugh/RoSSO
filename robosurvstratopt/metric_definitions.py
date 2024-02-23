import inspect
from inspect import signature

from icecream import ic
import jax
import jax.numpy as jnp
import numpy as np

import graph_comp
import strat_comp

#####################################################
# TODO: add support for differential metrics later...
def abs_P_diff_sum(P, problem_params):
    P_old = problem_params["P_old"]
    return jnp.sum(jnp.abs(P - P_old))

def abs_P_diff_max(P, problem_params):
    P_old = problem_params["P_old"]
    return jnp.max(jnp.abs(P - P_old))
#####################################################
# Below are some simple "absolute" metrics we can test with:
def lowest_cap_probs_range_Stackelberg(P, problem_params):
    F0 = problem_params["F0"]
    tau = problem_params["tau"]
    LCPs = strat_comp.compute_LCPs(P, F0, tau, num_LCPs=1)
    return jnp.ptp(LCPs)

def MCP_attack_node_index_Stackelberg(P, problem_params):
    F0 = problem_params["F0"]
    tau = problem_params["tau"]
    F = strat_comp.compute_cap_probs(P, F0, tau)
    return jnp.argmin(F, axis=1)

def MCP_robot_node_index_Stackelberg(P, problem_params):
    F0 = problem_params["F0"]
    tau = problem_params["tau"]
    F = strat_comp.compute_cap_probs(P, F0, tau)
    return jnp.argmin(F, axis=0)

def diam_pair_shortest_path_cap_probs_Stackelberg(A, P, F0, tau, my_arg=None, your_arg=True, our_arg=1.0):
    diam_pairs = graph_comp.get_diametric_pairs(A)
    FHT_mats = strat_comp.compute_FHT_probs(P, F0, tau)
    return strat_comp.compute_SPCPs(A, FHT_mats, diam_pairs)


METRICS_REGISTRY = {
    "LCPs_range_Stackelberg" : lowest_cap_probs_range_Stackelberg,
    "MCP_attacker_location" : MCP_attack_node_index_Stackelberg,
    "MCP_robot_location" : MCP_robot_node_index_Stackelberg,
    "DP_SPCPs_Stackelberg" : diam_pair_shortest_path_cap_probs_Stackelberg
}

