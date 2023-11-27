from collections import deque
import jax.numpy as jnp
import strat_comp

class ProblemSpec:
    def __init__(self, name, problem_params, opt_params):
        self.name = name
        self.problem_params = problem_params
        self.opt_params = opt_params
        self.cnvg_test_vals = deque()

    def set_learning_rate(self):
        init_grad = self.compute_gradient()
        lr_scale = jnp.max(jnp.abs(init_grad))
        lr = self.opt_params["nominal_learning_rate"]/lr_scale
        self.opt_params["scaled_learning_rate"] = lr

    def apply_parametrization(self, Q):
        if 'multi' in self.problem_params.obj_fun_flag:
            for i in range(N):
                P_i = strat_comp.comp_P_param(Q[i, :, :], As[i, :, :], self.opt_params["use_abs_param"])
                P = P.at[i, :, :].set(P_i)
        else:
            P = strat_comp.comp_P_param(Q, A, self.opt_params["use_abs_param"])
        return P
    
    def cnvg_check(self, iter, abs_P_diff_sum, MCP_diff, loss_diff):
        def cnvg_check_inner(iter, new_val):
            self.cnvg_test_vals.append(new_val)
            if iter > self.opt_params["cnvg_window_size"]:
                self.cnvg_test_vals.popleft()
            MA_val = jnp.mean(self.cnvg_test_vals)
            if iter > self.opt_params["cnvg_window_size"]:
                converged = MA_val < self.opt_params["cnvg_radius"]
            else:
                converged = False
            if iter + 1 == self.opt_params["max_iters"]:
                converged = True
            return converged

        if self.opt_params["cnvg_test_mode"] == "P_update":
            converged = cnvg_check_inner(iter, abs_P_diff_sum)
        elif self.opt_params["cnvg_test_mode"] == "MCP_diff":
            converged = cnvg_check_inner(iter, MCP_diff)
        elif self.opt_params["cnvg_test_mode"] == "loss_diff":
            converged = cnvg_check_inner(iter, loss_diff)
        return converged

    def compute_gradient(self):
        if self.problem_params.obj_fun_flag == 'Stackelberg':
            grad = strat_comp.comp_avg_LCP_grad(P0, A, F0, tau, opt_params["num_init_LCPs"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'Stackelberg_pi':
            grad = strat_comp.comp_avg_LCP_pi_grad(P0, A, F0, tau, pi, opt_params["alpha"], opt_params["num_init_LCPs"], opt_params["use_abs_param"]) 
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg':
            grad = strat_comp.comp_avg_weighted_LCP_grad(P0, A, D_idx, W, w_max, tau, opt_params["num_init_LCPs"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg_pi':
            grad = strat_comp.comp_avg_weighted_LCP_pi_grad(P0, A, D_idx, W, w_max, tau, pi, opt_params["alpha"], opt_params["num_init_LCPs"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg_co_opt':
            grad = strat_comp.comp_avg_greedy_co_opt_weighted_LCP_grad(P0, A, D_idx, W, w_max, B, opt_params["num_init_LCPs"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg_co_opt_pi':
            grad = strat_comp.comp_avg_greedy_co_opt_weighted_LCP_pi_grad(P0, A, D_idx, W, w_max, B, pi, opt_params["alpha"], opt_params["num_init_LCPs"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'multi_weighted_Stackelberg':
            grad = strat_comp.comp_avg_weighted_multi_LCP_grad(P0, As, D_idx, combs, N, combs_len, W, w_max, tau, opt_params["num_init_LCPs"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'multi_weighted_Stackelberg_pi':
            grad = strat_comp.comp_avg_weighted_multi_LCP_pi_grad(P0, As, D_idx, combs, N, combs_len, W, w_max, tau, pi, opt_params["alpha"], opt_params["num_init_LCPs"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'MHT':
            grad = strat_comp.comp_MHT_grad(P0, A, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'MHT_pi':
            grad = strat_comp.comp_MHT_pi_grad(P0, A, pi, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_MHT_pi':
            grad = strat_comp.comp_weighted_MHT_pi_grad(P0, A, W, pi, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'ER_pi':
            grad = strat_comp.comp_ER_pi_grad(P0, A, pi, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'RTE_pi':
            N_eta = int(jnp.ceil(1/(eta*jnp.min(jnp.array(pi)))) - 1)
            grad = strat_comp.comp_RTE_pi_grad(P0, A, pi, N_eta, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_RTE_pi':
            N_eta = int(jnp.ceil(w_max/(eta*jnp.min(jnp.array(pi)))) - 1)
            grad = strat_comp.comp_weighted_RTE_pi_grad(P0, A, D_idx, W, w_max, pi, N_eta, opt_params["alpha"], opt_params["use_abs_param"])
        return grad
    
    def compute_loss(self):
        if self.problem_params.obj_fun_flag == 'Stackelberg':
            loss = strat_comp.loss_LCP(Q, A, F0, tau, num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'Stackelberg_pi':
            loss = strat_comp.loss_LCP_pi(Q, A, F0, tau, pi, opt_params["alpha"], num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg':
            loss = strat_comp.loss_weighted_LCP(Q, A, D_idx, W, w_max, tau, num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg_pi':
            loss = strat_comp.loss_weighted_LCP_pi(Q, A, D_idx, W, w_max, tau, pi, opt_params["alpha"], num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg_co_opt':
            loss = strat_comp.loss_greedy_co_opt_weighted_LCP(Q, A, D_idx, W, w_max, B, num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_Stackelberg_co_opt_pi':
            loss = strat_comp.loss_greedy_co_opt_weighted_LCP_pi(Q, A, D_idx, W, w_max, B, pi, opt_params["alpha"], num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'multi_weighted_Stackelberg':
            loss = strat_comp.loss_weighted_multi_LCP(Q, As, D_idx, combs, N, combs_len, W, w_max, tau, num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'multi_weighted_Stackelberg_pi':
            loss = strat_comp.loss_weighted_multi_LCP_pi(Q, As, D_idx, combs, N, combs_len, W, w_max, tau, pi, opt_params["alpha"], num_LCPs, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'MHT':
            loss = strat_comp.loss_MHT(Q, A, opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'MHT_pi':
            loss = strat_comp.loss_MHT_pi(Q, A, pi, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_MHT_pi':
            loss = strat_comp.loss_weighted_MHT_pi(Q, A, W, pi, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'ER_pi':
            loss = strat_comp.loss_ER_pi(Q, A, pi, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'RTE_pi':
            loss = strat_comp.loss_RTE_pi(Q, A, pi, N_eta, opt_params["alpha"], opt_params["use_abs_param"])
        elif self.problem_params.obj_fun_flag == 'weighted_RTE_pi':
            loss = strat_comp.loss_weighted_RTE_pi(Q, A, D_idx, W, w_max, pi, N_eta, opt_params["alpha"], opt_params["use_abs_param"])
        return loss

    def init_rand_P(self):
        if 'multi' in self.problem_params.obj_fun_flag:
            P = strat_comp.multi_init_rand_Ps(As, N, num)
        else:
            P = strat_comp.init_rand_Ps(A, num)
        return P
    
# Example usage:

# Create a ProblemSpec with additional properties
problem_spec = ProblemSpec(
    name="PS1",
    problem_params={
            "num_init_Ps": 10,
            "trackers": [
            "iters",
            "P_diff_sums",
            "P_diff_max_elts",
            "MCP_inds",
            "MCPs",
            "final_MCP",
            "final_iters",
            "loss",
            "loss_diff",
            "final_loss",
            "final_penalty",
            "final_tauvec"
        ],
        "graph_names": {
            "test1": "complete_N12"
        },
        "objective_functions": {
            "test1": "weighted_MHT_pi"
        },
        "stationary_distributions": {
            "test1": [0.1536, 0.1039, 0.1028, 0.1005, 0.0958, 0.0958, 0.0855, 0.0739, 0.0554, 0.0497, 0.0439, 0.0392]
        },
        "taus": {
            "test1": -1
        },
        "defense_budgets":{
            "test1": -1
        },
        "etas": {
            "test1": -1
        },
        "graph_codes": {
            "test1": "N12_147573952589676412927"
        },
        "weight_matrices": {
            "test1": [[1, 3, 3, 5, 4, 6, 3, 5, 7, 4, 6, 6],
                    [3, 1, 5, 4, 2, 4, 4, 5, 5, 3, 5, 5],
                    [3, 5, 1, 7, 6, 8, 3, 4, 9, 4, 8, 7],
                    [6, 4, 7, 1, 5, 6, 4, 7, 5, 6, 6, 7],
                    [4, 3, 6, 5, 1, 3, 5, 5, 6, 3, 4, 4],
                    [6, 4, 8, 5, 3, 1, 6, 7, 3, 6, 2, 3],
                    [2, 5, 3, 5, 6, 7, 1, 5, 7, 5, 7, 8],
                    [3, 5, 2, 7, 6, 7, 3, 1, 9, 3, 7, 5],
                    [8, 6, 9, 4, 6, 4, 6, 9, 1, 8, 5, 7],
                    [4, 3, 4, 6, 3, 5, 5, 3, 7, 1, 5, 3],
                    [6, 4, 8, 6, 4, 2, 6, 6, 4, 5, 1, 3],
                    [6, 4, 6, 6, 3, 3, 6, 4, 5, 3, 2, 1]]
        },
        "num_robots": {
            "test1": -1
        }},
    opt_params={
            "num_init_LCPs": 4,
            "num_LCPs_schedule": {
                "0": 1
            },
            "use_edge_weights": True,
            "use_abs_param": True,
            "alpha": 1,
            "max_iters": 10000,
            "optimizer_name": "rmsprop",
            "nominal_learning_rate": 1,
            "lr_schedule_type": "constant",
            "lr_schedule_args": None,
            "use_momentum": False,
            "mom_decay_rate": 0.9,
            "use_nesterov": False,
            "cnvg_test_mode": "loss_diff", 
            "cnvg_radius": 0.01,
            "cnvg_window_size": 10,
            "iters_per_printout": 500,
            "iters_per_trackvals": 10
        }
)
