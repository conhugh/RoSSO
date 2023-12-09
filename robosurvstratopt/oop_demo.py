from collections import deque
import jax.numpy as jnp
import numpy as np
import strat_comp
import graph_comp

class ProblemSpec:
    def __init__(self, name, problem_params, opt_params):
        self.name = name
        self.problem_params = problem_params
        self.opt_params = opt_params

    def initialize(self):
        self.cnvg_test_vals = deque()
        self.problem_params["adjacency_matrix"] = graph_comp.graph_decode(self.problem_params["graph_code"])
        self.n = self.problem_params["adjacency_matrix"].shape[0]
        if 'pi' in self.problem_params["objective_function"]:
            self.pi = tuple(self.problem_params["stationary_distribution"])
        if 'weighted' in self.problem_params["objective_function"]:
            self.problem_params["weight_matrix"] = jnp.array(self.problem_params["weight_matrix"])
            self.w_max = int(jnp.max(self.problem_params["weight_matrix"]))
        if 'multi' in self.problem_params["objective_function"]:
            self.combs, self.combs_len = strat_comp.precompute_multi(self.n, self.problem_params["num_robots"])
        if 'Stackelberg' in self.problem_params["objective_function"]:
            self.F0 = jnp.full((self.n, self.n, self.problem_params["tau"]), jnp.nan)
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

    def set_learning_rate(self, Q):
        init_grad = self.compute_gradient(Q)
        lr_scale = jnp.max(jnp.abs(init_grad))
        lr = self.opt_params["nominal_learning_rate"]/lr_scale
        self.opt_params["scaled_learning_rate"] = lr

    def apply_parametrization(self, Q):
        if 'multi' in self.problem_params["objective_function"]:
            for i in range(self.problem_params["num_robots"]):
                P_i = strat_comp.comp_P_param(Q[i, :, :], self.problem_params["adjacency_matrix"][i, :, :], self.opt_params["use_abs_param"])
                P = Q.at[i, :, :].set(P_i)
        else:
            P = strat_comp.comp_P_param(Q, self.problem_params["adjacency_matrix"], self.opt_params["use_abs_param"])
        return P
    
    def cnvg_check(self, iter, abs_P_diff_sum, loss_diff):
        def cnvg_check_inner(iter, new_val):
            self.cnvg_test_vals.append(new_val)
            if iter > self.opt_params["cnvg_window_size"]:
                self.cnvg_test_vals.popleft()
            MA_val = np.mean(self.cnvg_test_vals)
            if iter > self.opt_params["cnvg_window_size"]:
                converged = MA_val < self.opt_params["cnvg_radius"]
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

    def compute_gradient(self, Q):
        if self.problem_params["objective_function"] == 'Stackelberg':
            grad = strat_comp.comp_avg_LCP_grad(Q, self.problem_params["adjacency_matrix"], self.F0, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'Stackelberg_pi':
            grad = strat_comp.comp_avg_LCP_pi_grad(Q, self.problem_params["adjacency_matrix"], self.F0, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"]) 
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg':
            grad = strat_comp.comp_avg_weighted_LCP_grad(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_pi':
            grad = strat_comp.comp_avg_weighted_LCP_pi_grad(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_co_opt':
            grad = strat_comp.comp_avg_greedy_co_opt_weighted_LCP_grad(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["B"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_co_opt_pi':
            grad = strat_comp.comp_avg_greedy_co_opt_weighted_LCP_pi_grad(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["B"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'multi_weighted_Stackelberg':
            grad = strat_comp.comp_avg_weighted_multi_LCP_grad(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.combs, self.problem_params["num_robots"], self.combs_len, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'multi_weighted_Stackelberg_pi':
            grad = strat_comp.comp_avg_weighted_multi_LCP_pi_grad(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.combs, self.problem_params["num_robots"], self.combs_len, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'MHT':
            grad = strat_comp.comp_MHT_grad(Q, self.problem_params["adjacency_matrix"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'MHT_pi':
            grad = strat_comp.comp_MHT_pi_grad(Q, self.problem_params["adjacency_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_MHT_pi':
            grad = strat_comp.comp_weighted_MHT_pi_grad(Q, self.problem_params["adjacency_matrix"], self.problem_params["weight_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'ER_pi':
            grad = strat_comp.comp_ER_pi_grad(Q, self.problem_params["adjacency_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'RTE_pi':
            grad = strat_comp.comp_RTE_pi_grad(Q, self.problem_params["adjacency_matrix"], self.pi, self.problem_params["N_eta"], self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_RTE_pi':
            grad = strat_comp.comp_weighted_RTE_pi_grad(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.pi, self.problem_params["N_eta"], self.opt_params["alpha"], self.opt_params["use_abs_param"])
        return grad
    
    def compute_loss(self, Q):
        if self.problem_params["objective_function"] == 'Stackelberg':
            loss = strat_comp.loss_LCP(Q, self.problem_params["adjacency_matrix"], self.F0, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'Stackelberg_pi':
            loss = strat_comp.loss_LCP_pi(Q, self.problem_params["adjacency_matrix"], self.F0, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg':
            loss = strat_comp.loss_weighted_LCP(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_pi':
            loss = strat_comp.loss_weighted_LCP_pi(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_co_opt':
            loss = strat_comp.loss_greedy_co_opt_weighted_LCP(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["B"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_Stackelberg_co_opt_pi':
            loss = strat_comp.loss_greedy_co_opt_weighted_LCP_pi(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.problem_params["B"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'multi_weighted_Stackelberg':
            loss = strat_comp.loss_weighted_multi_LCP(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.combs, self.problem_params["num_robots"], self.combs_len, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'multi_weighted_Stackelberg_pi':
            loss = strat_comp.loss_weighted_multi_LCP_pi(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.combs, self.problem_params["num_robots"], self.combs_len, self.problem_params["weight_matrix"], self.w_max, self.problem_params["tau"], self.pi, self.opt_params["alpha"], self.opt_params["num_LCPs"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'MHT':
            loss = strat_comp.loss_MHT(Q, self.problem_params["adjacency_matrix"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'MHT_pi':
            loss = strat_comp.loss_MHT_pi(Q, self.problem_params["adjacency_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_MHT_pi':
            loss = strat_comp.loss_weighted_MHT_pi(Q, self.problem_params["adjacency_matrix"], self.problem_params["weight_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'ER_pi':
            loss = strat_comp.loss_ER_pi(Q, self.problem_params["adjacency_matrix"], self.pi, self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'RTE_pi':
            loss = strat_comp.loss_RTE_pi(Q, self.problem_params["adjacency_matrix"], self.pi, self.problem_params["N_eta"], self.opt_params["alpha"], self.opt_params["use_abs_param"])
        elif self.problem_params["objective_function"] == 'weighted_RTE_pi':
            loss = strat_comp.loss_weighted_RTE_pi(Q, self.problem_params["adjacency_matrix"], self.D_idx, self.problem_params["weight_matrix"], self.w_max, self.pi, self.problem_params["N_eta"], self.opt_params["alpha"], self.opt_params["use_abs_param"])
        return loss

    def init_rand_Ps(self):
        if 'multi' in self.problem_params["objective_function"]:
            self.problem_params["adjacency_matrix"] = jnp.tile(self.problem_params["adjacency_matrix"], (self.problem_params["num_robots"], 1, 1))
            P = strat_comp.multi_init_rand_Ps(self.problem_params["adjacency_matrix"], self.problem_params["num_robots"], self.opt_params["num_init_Ps"])
        else:
            P = strat_comp.init_rand_Ps(self.problem_params["adjacency_matrix"], self.opt_params["num_init_Ps"])
        return P
    
