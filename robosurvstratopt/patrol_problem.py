from collections import deque
import os
import jax
import jax.numpy as jnp
import numpy as np

import graph_comp
from metric_tracker import MetricTracker
import strat_comp

class PatrolProblem:
    def __init__(self, name, problem_params, opt_params, results_directory):
        self.name = name
        self.results_dir = results_directory
        self.problem_params = problem_params
        self.opt_params = opt_params
        self.metrics = [MetricTracker(m_name) for m_name in opt_params["metrics"]]

    def initialize(self):
        self.key = jax.random.PRNGKey(self.opt_params["rng_seed"])
        self.cnvg_test_vals = deque()
        self.cnvg_MA_val = 0
        self.problem_params["adjacency_matrix"] = graph_comp.graph_decode(self.problem_params["graph_code"])
        self.n = self.problem_params["adjacency_matrix"].shape[0]
        if 'pi' in self.problem_params["objective_function"]:
            self.pi = tuple(self.problem_params["stationary_distribution"])
        if 'weighted' in self.problem_params["objective_function"]:
            self.problem_params["weight_matrix"] = jnp.array(self.problem_params["weight_matrix"])
            self.w_max = int(jnp.max(self.problem_params["weight_matrix"]))
            if self.problem_params["tau"] < self.w_max:
                raise ValueError("tau is less than the maximum travel time!")
        if self.problem_params["num_robots"] > 1 and 'multi' not in self.problem_params["objective_function"]:
            raise ValueError("num_robots is greater than 1 and the objective function is not a multi-robot objective!")
        if 'multi' in self.problem_params["objective_function"]:
            self.combs, self.combs_len = strat_comp.precompute_multi(self.n, self.problem_params["num_robots"])
        if 'multi_Stackelberg' in self.problem_params["objective_function"]:
            self.problem_params["F0"] = jnp.full((self.problem_params["num_robots"], self.n, self.n, self.problem_params["tau"]), jnp.nan)
        elif 'Stackelberg' in self.problem_params["objective_function"]:
            self.problem_params["F0"] = jnp.full((self.n, self.n, self.problem_params["tau"]), jnp.nan)
        if 'RTE' in self.problem_params["objective_function"]:
            if 'weighted' in self.problem_params["objective_function"]:
                self.problem_params["N_eta"] = int(jnp.ceil(self.w_max/(self.problem_params["eta"]*jnp.min(jnp.array(self.pi)))) - 1)
            else:
                self.problem_params["N_eta"] = int(jnp.ceil(1/(self.problem_params["eta"]*jnp.min(jnp.array(self.pi)))) - 1)
        if 'weighted_Stackelberg_co_opt' in self.problem_params["objective_function"]:
            self.D_idx = strat_comp.precompute_weighted_Stackelberg_co_opt(self.problem_params["weight_matrix"], self.w_max, self.problem_params["B"])
        elif 'weighted_Stackelberg' in self.problem_params["objective_function"]:
            self.D_idx = strat_comp.precompute_weighted_Stackelberg(self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"])
        elif 'weighted_RTE_pi' in self.problem_params["objective_function"]:
            self.D_idx = strat_comp.precompute_weighted_RTE_pi(self.problem_params["weight_matrix"], self.w_max, self.problem_params["N_eta"])

    def update_metrics_tracking(self, P):
        for metric in self.metrics:
            metric.update_history(P, self.problem_params)

    def save_tracked_metrics(self, save_directory=None):
        if save_directory is None:
            save_directory = os.path.join(self.results_dir, "metrics")
        for metric in self.metrics:
            metric.print_history(save_directory)

    def set_learning_rate(self, Q):
        (_, init_grad) = self.compute_loss_and_gradient(Q)
        lr_scale = jnp.max(jnp.abs(init_grad))
        lr = self.opt_params["nominal_learning_rate"]/lr_scale
        self.opt_params["scaled_learning_rate"] = lr

    def apply_parametrization(self, Q):
        if 'multi' in self.problem_params["objective_function"]:
            for i in range(self.problem_params["num_robots"]):
                P_i = strat_comp.comp_P_param(Q[i, :, :], self.problem_params["adjacency_matrix"][i, :, :], self.opt_params["use_abs_param"])
                P = Q.at[i, :, :].set(P_i)
        else:
            P = strat_comp.comp_P_param(Q[0], self.problem_params["adjacency_matrix"], self.opt_params["use_abs_param"])
        return P
    
    def cnvg_check(self, iter, abs_P_diff_sum, loss_diff):
        def cnvg_check_inner(iter, new_val):
            self.cnvg_test_vals.append(new_val)
            # if iter >= self.opt_params["cnvg_window_size"]:
            #     self.cnvg_MA_val = self.cnvg_MA_val + ((new_val - self.cnvg_test_vals.popleft())/self.opt_params["cnvg_window_size"])
            # else:
            #     self.cnvg_MA_val = np.mean(self.cnvg_test_vals)
            if iter >= self.opt_params["cnvg_window_size"]:
                self.cnvg_test_vals.popleft()
            self.cnvg_MA_val = np.mean(self.cnvg_test_vals)
            if iter > self.opt_params["cnvg_window_size"]:
                converged = self.cnvg_MA_val < self.opt_params["cnvg_radius"]
            else:
                converged = False
            if iter + 1 == self.opt_params["max_iters"]:
                converged = True
            return converged

        if self.opt_params["cnvg_test_mode"] == "P_update":
            converged = cnvg_check_inner(iter, abs_P_diff_sum)
        # elif self.opt_params["cnvg_test_mode"] == "MCP_diff":
        #     converged = cnvg_check_inner(iter, MCP_diff)
        elif self.opt_params["cnvg_test_mode"] == "loss_diff":
            converged = cnvg_check_inner(iter, loss_diff)
        return converged
    
    def compute_loss_and_gradient(self, Q):
        if self.problem_params["objective_function"] == 'Stackelberg':
            out = jax.value_and_grad(strat_comp.loss_LCP)(Q, self.problem_params["adjacency_matrix"], self.problem_params["F0"], self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'Stackelberg_pi':
            out = jax.value_and_grad(strat_comp.loss_LCP_pi)(Q, self.problem_params["adjacency_matrix"], self.problem_params["F0"], self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg':
            out = jax.value_and_grad(strat_comp.loss_weighted_LCP)(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_pi':
            out = jax.value_and_grad(strat_comp.loss_weighted_LCP_pi)(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_co_opt':
            out = jax.value_and_grad(strat_comp.loss_greedy_co_opt_weighted_LCP)(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["B"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_co_opt_pi':
            out = jax.value_and_grad(strat_comp.loss_greedy_co_opt_weighted_LCP_pi)(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["B"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'multi_Stackelberg':
            out = jax.value_and_grad(strat_comp.loss_multi_LCP)(Q, self.problem_params["adjacency_matrix"], self.problem_params["F0"], self.combs, self.problem_params["num_robots"], self.combs_len, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'multi_weighted_Stackelberg':
            out = jax.value_and_grad(strat_comp.loss_weighted_multi_LCP)(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.combs, self.problem_params["num_robots"], self.combs_len, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'multi_weighted_Stackelberg_pi':
            out = jax.value_and_grad(strat_comp.loss_weighted_multi_LCP_pi)(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.combs, self.problem_params["num_robots"], self.combs_len, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'MHT':
            out = jax.value_and_grad(strat_comp.loss_MHT)(Q, self.problem_params["adjacency_matrix"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'MHT_pi':
            out = jax.value_and_grad(strat_comp.loss_MHT_pi)(Q, self.problem_params["adjacency_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_MHT_pi':
            out = jax.value_and_grad(strat_comp.loss_weighted_MHT_pi)(Q, self.problem_params["adjacency_matrix"], self.problem_params["weight_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'ER_pi':
            out = jax.value_and_grad(strat_comp.loss_ER_pi)(Q, self.problem_params["adjacency_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'RTE_pi':
            out = jax.value_and_grad(strat_comp.loss_RTE_pi)(Q, self.problem_params["adjacency_matrix"], self.pi, self.problem_params["N_eta"], self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_RTE_pi':
            out = jax.value_and_grad(strat_comp.loss_weighted_RTE_pi)(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.pi, self.problem_params["N_eta"], self.opt_params["alpha"], self.opt_params["use_abs_param"])
        return out

    def init_rand_Ps(self):
        # current implementation assumes same adjacency matrix for each robot
        if self.problem_params["num_robots"] > 1:
            self.problem_params["adjacency_matrix"] = jnp.tile(self.problem_params["adjacency_matrix"], (self.problem_params["num_robots"], 1, 1))
        P = strat_comp.oop_init_rand_Ps(self.problem_params["adjacency_matrix"], self.problem_params["num_robots"], self.opt_params["num_init_Ps"], self.key)
        return P
    
