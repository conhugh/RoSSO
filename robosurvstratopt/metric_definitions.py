import inspect
from inspect import signature

from icecream import ic
import jax
import jax.numpy as jnp
import numpy as np

import graph_comp
import strat_comp


def min_cap_prob_Stackelberg(P, F0, tau):
    return strat_comp.compute_LCPs(P, F0, tau, num_LCPs=1)

# def abs_P_diff_sum(P, P_old):
def abs_P_diff_sum(Q, P, loss, Q_old, P_old, loss_old, problem_params):
    return jnp.sum(jnp.abs(P - P_old))



# def abs_P_diff_sum(P, P_old):
    # return jnp.sum(jnp.abs(P - P_old))

def abs_P_diff_max(P, P_old):
    return jnp.max(jnp.abs(P - P_old))


def min_cap_prob_idx_Stackelberg(P, problem_params):

    F0 = problem_params["F0"]
    tau = problem_params["tau"]
    return strat_comp.compute_LCPs(P, F0, tau, num_LCPs=1)


def test_func(P, A, w=None, h=False):
    print(P)
    print(A)
    print(w)
    print(h)

# def lowest_cap_probs_Stackelberg(P, F0, tau, num_LCPs):
    # return strat_comp.compute_LCPs(P, F0, tau, num_LCPs=num_LCPs)


# def diam_pair_shortest_path_cap_probs_Stackelberg(A, P, F0, tau, my_arg=None, your_arg=True, our_arg=1.0):
#     diam_pairs = graph_comp.get_diametric_pairs(A)
#     # F = strat_comp.compute_FHT_probs(P, F0, tau)
#     # return strat_comp.compute_SPCPs(A, F, diam_pairs)
#     return diam_pairs


def test_print_func(f):
    print(f.__name__)

METRICS_REGISTRY = {
    "abs_P_diff_sum" : abs_P_diff_sum,
    "abs_P_diff_max" : abs_P_diff_max,
    "MCP_Stackelberg" : min_cap_prob_Stackelberg,
    # "LCPs_Stackelberg" : lowest_cap_probs_Stackelberg,
    # "DP_SPCPs_Stackelberg" : diam_pair_shortest_path_cap_probs_Stackelberg
}

if __name__ == '__main__':
    pass
    # ic(signature(diam_pair_shortest_path_cap_probs_Stackelberg))
    # ic(signature(diam_pair_shortest_path_cap_probs_Stackelberg).parameters)
    # ic(list(signature(diam_pair_shortest_path_cap_probs_Stackelberg).parameters.keys()))
    # test_print_func(diam_pair_shortest_path_cap_probs_Stackelberg)
    # ic(inspect.getfullargspec(diam_pair_shortest_path_cap_probs_Stackelberg))

#     test_args_list = ["C", True, "green", 4]
#     test_func(*test_args_list)

    

