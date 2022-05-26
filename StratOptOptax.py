# Optimization of the performance of stochastic surveillance strategies
import itertools
from pydoc import doc
import numpy as np
import os, shutil
import time
import json
from collections import deque
import optax
import jax.numpy as jnp
import TestSpec as ts
from StratCompJax import *
from StratViz import *

def test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters):
    """
    Test and time various Optax optimizers for a fixed number of iterations.

    Parameters
    ----------
    P0 : numpy.ndarray 
        Random initial transition probability matrix. 
    F0 : numpy.ndarray
        Placeholder for storing FHT probability matrices. 
    tau : int
        Intruder's attack duration. 
    eps0 : float
        Learning rate.
    max_iters : int
        Total number of iterations.
    
    Returns
    -------
    numpy.ndarray
        Transition probability matrix corresponding to optimized strategy.
    numpy.ndarray
        Capture Probability matrix corresponding to optimized strategy.
    """
    init_Ps = init_rand_Ps(A, num_init_Ps)
    num_LCPs = 4
    nominal_learning_rate = 0.0001
    grad_mode = "MCP_parametrization"
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)

    for k in range(num_init_Ps):
        print("---------------------Optimizing with initial P matrix number " + str(k + 1) + "----------------------------")
        P0 = init_Ps[:, :, k]
        P = P0
        Q = P0
        # print("P0 shape = " + str(jnp.shape(P0)) + ", F0 shape = " + str(jnp.shape(F0)) + ", A shape = " + str(jnp.shape(A)))
        scaled_learning_rate, _ = set_learning_rate(P0, A, F0, tau, num_LCPs, nominal_learning_rate, grad_mode)
        print("scaled_learning_rate = " + str(scaled_learning_rate))
        # optimizer = optax.rmsprop(scaled_learning_rate)
        optimizer = optax.rmsprop(scaled_learning_rate, momentum=0.9, nesterov=False)
        # optimizer = optax.sgd(scaled_learning_rate, momentum=0.99, nesterov=True)
        opt_state = optimizer.init(P0)
        check_time = time.time()
        for iter in range(max_iters + 1):
            # bound the update to the P matrix:
            # updates = jnp.minimum(updates, P_update_bound)
            # updates = jnp.maximum(updates, -1*P_update_bound)
            # print("iteration number: " + str(k))
            # start_time = time.time()
            if grad_mode == "MCP_parametrization":
                grad = -1*comp_MCP_grad_param_abs(Q, A, F0, tau) # negate so that the optimizer does gradient ascent
            elif grad_mode == "MCP_abs_parametrization":
                grad = -1*comp_MCP_grad_param_abs(Q, A, F0, tau) # negate so that the optimizer does gradient ascent
            elif grad_mode == "LCP_parametrization":
                grad = -1*comp_avg_LCP_grad_param(Q, A, F0, tau, num_LCPs) # negate so that the optimizer does gradient ascent
            elif grad_mode == "MCP_projection":
                grad = -1*comp_MCP_grad(P, F0, tau) # negate so that the optimizer does gradient ascent
            elif grad_mode == "LCP_projection":
                grad = -1*comp_avg_LCP_grad(P, F0, tau, num_LCPs) # negate so that the optimizer does gradient ascent
            else:
                raise ValueError("Invalid grad_mode specified!")
            # print("--- Getting grad took: %s seconds ---" % (time.time() - start_time))
            # start_time = time.time()
            updates, opt_state = optimizer.update(grad, opt_state)
            # print("--- Getting update and state took: %s seconds ---" % (time.time() - start_time))
            # start_time = time.time()
            if grad_mode.find("parametrization") != -1:
                Q = optax.apply_updates(Q, updates)
                P = comp_P_param(Q, A)
            else:
                P = optax.apply_updates(P, updates)
                P = proj_onto_simplex(P)
            # print("--- Applying update took: %s seconds ---" % (time.time() - start_time))
            if iter % 200 == 0:
            # if iter % 1 == 0: 
                print("------ iteration number " + str(iter) + ", elapsed time =  " + str(time.time() - check_time)[:7] + "-------")
                print("grad 1-norm = " + str(jnp.sum(jnp.abs(grad))))
                print("grad inf norm = " + str(jnp.max(jnp.abs(grad))))
                print("MCP = " + str(compute_MCP(P, F0, tau)))


        print("Final MCP:")
        print(compute_MCP(P, F0, tau))
        
        


# def sgd_grad_conv(P0, A, F0, tau, opt_params, trackers):
#     check_time = time.time()
#     n = P0.shape[0]
#     P = P0 
#     Q = P0
#     if opt_params["use_momentum"]:
#         optimizer = optax.sgd(opt_params["scaled_learning_rate"], momentum=opt_params["mom_decay_rate"], nesterov=opt_params["use_nesterov"])
#     else:
#         optimizer = optax.sgd(opt_params["scaled_learning_rate"])

#     opt_state = optimizer.init(P0)
#     rec_P_diff_sums = deque()  # queue storing recent P matrices, for checking convergence
#     ub_P_update = jnp.full((n, n), opt_params["P_update_elt_bound"])
#     lb_P_update = jnp.full((n, n), -1*opt_params["P_update_elt_bound"])
#     tracked_vals = {track_val : [] for track_val in trackers}

#     iter = 0  # number of gradient ascent steps taken so far
#     converged = False
#     while not converged:
#         # take gradient ascent step:
#         if opt_params["grad_mode"] == "MCP_parametrization":
#             grad = -1*comp_MCP_grad_param(Q, A, F0, tau) # negate so that the optimizer does gradient ascent
#         elif opt_params["grad_mode"] == "LCP_parametrization":
#             grad = -1*comp_avg_LCP_grad_param(Q, A, F0, tau, opt_params["num_LCPs"]) # negate so that the optimizer does gradient ascent
#         elif opt_params["grad_mode"] == "MCP_projection":
#             grad = -1*comp_MCP_grad(P, F0, tau) # negate so that the optimizer does gradient ascent
#         elif opt_params["grad_mode"] == "LCP_projection":
#             grad = -1*comp_avg_LCP_grad(P, F0, tau, opt_params["num_LCPs"]) # negate so that the optimizer does gradient ascent
#         else:
#             raise ValueError("Invalid grad_mode specified!")

#         updates, opt_state = optimizer.update(grad, opt_state)
#         # bound the update to the P matrix:
#         updates = jnp.minimum(updates, ub_P_update)
#         updates = jnp.maximum(updates, lb_P_update)

#         # apply update to P matrix and compute P_diff matrix:
#         if opt_params["grad_mode"].find("parametrization") != -1:
#             P_old = comp_P_param(Q, A)
#             Q = optax.apply_updates(Q, updates)
#             P = comp_P_param(Q, A)
#         else:
#             P_old = P
#             P = optax.apply_updates(P, updates)
#             P = proj_onto_simplex(P)
    
#         P_diff = P - P_old
#         # Compute sum of abs val of elements of P_diff and update queue of recent abs_P_diff_sums:
#         abs_P_diff_sum = jnp.sum(jnp.abs(P_diff))
#         rec_P_diff_sums.append(abs_P_diff_sum)
#         # compute updated moving average of recent abs_P_diff_sums:
#         if iter > opt_params["num_rec_P_diffs"]:
#             oldest_P_diff_sum = rec_P_diff_sums.popleft() 
#             new_MAP_diff_sum = (old_MAP_diff_sum*opt_params["num_rec_P_diffs"] - oldest_P_diff_sum + abs_P_diff_sum)/opt_params["num_rec_P_diffs"]
#         else:
#             new_MAP_diff_sum = np.mean(rec_P_diff_sums)
#         old_MAP_diff_sum = new_MAP_diff_sum

#         # track metrics of interest:
#         if(iter % opt_params["iters_per_trackvals"] == 0):
#             tracked_vals["iters"].append(iter)
#             tracked_vals["P_diff_sums"].append(abs_P_diff_sum)
#             tracked_vals["P_diff_max_elts"].append(jnp.max(jnp.abs(P_diff)))
#             F = compute_cap_probs(P, F0, tau).block_until_ready()
#             F = F.reshape((n**2), order='F')
#             tracked_vals["MCP_inds"].append(jnp.argmin(F))
#             tracked_vals["MCPs"].append(jnp.min(F))
#             # print status update to terminal:
#             if(iter % opt_params["iters_per_printout"] == 0):
#                 print("------ iteration number " + str(iter) + ", elapsed time =  " + str(time.time() - check_time) + "-------")
#                 print("grad 1-norm = " + str(jnp.sum(jnp.abs(grad))))
#                 print("grad inf norm = " + str(jnp.max(jnp.abs(grad))))
#                 print("abs_P_diff_sum = " + str(jnp.sum(jnp.abs(P_diff))))
#                 print("MCP = " + str(jnp.min(F)))

#         # check for element-wise convergence, update moving average abs_P_diff_sum and iteration counter:
#         if(iter > opt_params["num_rec_P_diffs"]):
#             converged = new_MAP_diff_sum < opt_params["radius"]
#         if iter == opt_params["max_iters"]:
#             converged = True
#         iter = iter + 1

#     # # convergence or max iteration count reached...
#     F = compute_cap_probs(P, F0, tau).block_until_ready()
#     final_MCP = jnp.min(F)
#     tracked_vals["final_MCP"].append(final_MCP)
#     print("Minimum Capture Probability at iteration " + str(iter) + ":")
#     print(final_MCP)
#     tracked_vals["final_iters"].append(iter)
#     return P, F, tracked_vals

# # Explore optima for the given graph and attack duration:
# def explore_graph_optima(A, tau, test_set_name, graph_num, grad_mode, save=True):
#     opt_params = {
#         "num_init_Ps" : 5,
#         "radius" : 0.005,
#         "num_rec_P_diffs" : 200,
#         "P_update_elt_bound" : 0.05,
#         "eps0" : 0.05,
#         "scaled_learning_rate" : None,
#         "use_momentum" : True,
#         "mom_decay_rate" : 0.99,
#         "use_nesterov" : True,
#         "max_iters" : 20000,
#         "grad_mode" : "MCP_parametrization",
#         "use_P_update_bound_schedule" : False,
#         "use_learning_rate_schedule" : False,
#         "use_num_LCPs_schedule" : False,
#         "num_LCPs" : 4,
#         "iters_per_printout" : 200,
#         "iters_per_trackvals" : 10
#         # "num_LCPs" : int(np.ceil((n**2)/10))
#     }

#     n = A.shape[0]
#     F0 = jnp.full((n, n, tau), np.NaN)

#     num_init_Ps = opt_params["num_init_Ps"]
#     init_Ps = init_rand_Ps(A, num_init_Ps)
#     learning_rates = np.zeros(num_init_Ps, dtype='float32')
#     lr_scales = np.zeros(num_init_Ps, dtype='float32')
#     conv_iters = np.zeros(num_init_Ps, dtype='float32')
#     conv_times = np.zeros(num_init_Ps, dtype='float32')
#     final_MCPs = np.zeros(num_init_Ps, dtype='float32')

#     print("num_LCPs = " + str(opt_params["num_LCPs"]))

#     if(save):
#         # Create directory for saving the results for the given graph:
#         project_dir = os.getcwd()
#         results_dir = os.path.join(project_dir, "Results/test_set_" + str(test_set_name))
#         if not os.path.isdir(results_dir):
#             os.mkdir(results_dir)
#         graph_name = "graph" + str(graph_num) + "_tau" + str(tau)
#         graph_dir = os.path.join(results_dir, graph_name)
#         if not os.path.isdir(graph_dir):
#             os.mkdir(graph_dir)

#     # Run the optimization algorithm:
#     for k in range(num_init_Ps):
#         print("Optimizing with initial P matrix number " + str(k + 1) + "...")
#         P0 = init_Ps[:, :, k]
#         lr, lr_scale = set_learning_rate(P0, A, F0, tau, opt_params["num_LCPs"], opt_params["eps0"], grad_mode)
#         lr_scales[k] = lr_scale
#         opt_params["scaled_learning_rate"] = lr
#         learning_rates[k] = lr
#         # --------------------------------------------------------------------
#         start_time = time.time()
#         P, F, tracked_vals = sgd_grad_conv(P0, A, F0, tau, opt_params)
#         conv_time = time.time() - start_time
#         print("--- Optimization took: %s seconds ---" % (conv_time))
#         # --------------------------------------------------------------------
#         conv_times[k] = conv_time
#         conv_iters[k] = tracked_vals["final_iters"][0]
#         final_MCPs[k] = tracked_vals["final_MCP"][0]
        
#         if(save):
#             # Save initial and optimized P matrices:
#             np.savetxt(graph_dir + "/init_P_" + str(k + 1) + ".csv", P0, delimiter=',')
#             np.savetxt(graph_dir + "/opt_P_" + str(k + 1) + ".csv", P, delimiter=',')
#             metrics_dir = os.path.join(graph_dir, "metrics")
#             if not os.path.isdir(metrics_dir):
#                 os.mkdir(metrics_dir)
#             # Save metrics tracked during the optimization process:
#             for metric in tracked_vals.keys():
#                 if metric.find("final") == -1:
#                     np.savetxt(metrics_dir + "/" + metric + "_" + str(k + 1) + ".csv", tracked_vals[metric], delimiter=',')


#     if(save):
#         np.savetxt(graph_dir + "/A.csv", A, delimiter=',')  # Save the env graph binary adjacency matrix
#         draw_env_graph(A, graph_num, graph_dir)  # Save a drawing of the env graph
#         graph_code = gen_graph_code(A)  # Generate unique graph code
#         # Write info file with graph and optimization algorithm info:
#         info_path = os.path.join(graph_dir, "info.txt")
#         with open(info_path, 'w') as info:
#             info.write("---------- Graph Information ----------\n")
#             info.write("Number of nodes (n) = " + str(n) + "\n")
#             info.write("Attack Duration (tau) = " + str(tau) + "\n")
#             info.write("Graph Code = " + graph_code + "\n")
#             info.write("---------- Optimizer Information ----------\n")
#             info.write("Optimizer used = sgd_grad_conv\n")
#             info.write("Gradient Mode = " + opt_params["grad_mode"] + "\n")
#             info.write("Number of LCPs used, if applicable = " + str(opt_params["num_LCPs"]) + "\n")
#             info.write("Bound on abs val of elements of update to P matrix = " + str(opt_params["P_update_elt_bound"]) + "\n")
#             info.write("Learning Rate = " + str(opt_params["scaled_learning_rate"]) + "\n")
#             info.write("eps0 = " + str(opt_params["eps0"]) + "\n")
#             info.write("Momentum Decay Rate = " + str(opt_params["mom_decay_rate"]) + "\n")
#             info.write("Nesterov = " + str(opt_params["use_nesterov"]) + "\n")
#             info.write("Convergence radius = " + str(opt_params["radius"]) + "\n")
#             info.write("Moving average window size (num_rec_P_diffs) = " + str(opt_params["num_rec_P_diffs"]) + "\n")
#             info.write("Max allowed number of iterations (max_iters) = " + str(opt_params["max_iters"]) + "\n")
#             info.write("Learning Rates = " + str(learning_rates) + "\n")
#             info.write("Max element of initial MCP gradients (lr_scales) = " + str(lr_scales) + "\n")
#             info.write("Final MCP achieved = " + str(final_MCPs) + "\n")
#             info.write("Number of iterations required = " + str(conv_iters) + "\n")
#             info.write("Optimization Times Required (seconds) = " + str(conv_times) + "\n")
#         info.close()


def set_learning_rate(P0, A, F0, tau, num_LCPs, nominal_LR, grad_mode):
    if grad_mode == "MCP_projection":
        init_grad = comp_MCP_grad(P0, A, F0, tau)
    elif grad_mode == "MCP_parametrization":
        init_grad = comp_MCP_grad_param(P0, A, F0, tau)
    elif grad_mode == "MCP_abs_parametrization":
        init_grad = comp_MCP_grad_param_abs(P0, A, F0, tau) 
    elif grad_mode == "LCP_parametrization":
        init_grad = comp_avg_LCP_grad_param(P0, A, F0, tau, num_LCPs)
    elif grad_mode == "LCP_projection":
        init_grad = comp_avg_LCP_grad(P0, A, F0, tau, num_LCPs)
    else:
        raise ValueError("Invalid grad_mode specified!")
    LR_scale = jnp.max(jnp.abs(init_grad))
    LR = nominal_LR/LR_scale
    return LR, LR_scale


# Run the set of tests described by test_specification:
def run_test_set(test_set_name, test_spec=None, save=True, opt_comparison=False):
    if type(test_spec) != ts.TestSpec:
        raise ValueError("Must provide a TestSpec object for test specification.")
    else:
        test_spec.validate_test_spec()

    if(save):
        project_dir = os.getcwd()
        test_set_dir = os.path.join(project_dir, "Results/test_set_" + str(test_set_name))
        if os.path.isdir(test_set_dir):
            input("WARNING! Results from a test with the same name have been saved previously, press ENTER to overwrite existing results.")
            for files in os.listdir(test_set_dir):
                path = os.path.join(test_set_dir, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)
        if not os.path.isdir(test_set_dir):
            os.mkdir(test_set_dir)
        test_spec.save_test_spec("test_spec", test_set_dir)

    run_times = []
    final_iters = []
    final_MCPs = []
    for test_num in range(1, test_spec.num_tests + 1):
        test_name = "test" + str(test_num)
        graph_name = test_spec.graph_names[test_name]
        graph_code = test_spec.graph_codes[test_name]
        A = graph_decode(graph_code)
        tau = test_spec.taus[test_name]
        trackers = test_spec.trackers
        if test_spec.optimizer_params["varying_optimizer_params"]:
            opt_params = test_spec.optimizer_params[test_name].copy()
        else:
            opt_params = test_spec.optimizer_params["params"].copy()
        if test_spec.schedules["varying_schedules"]:
            schedules = test_spec.schedules[test_name]
        else:
            schedules = test_spec.schedules["schedules"]
        print("-------- Working on Graph " + graph_name + " with tau = " + str(tau) + "----------")
        times, iters, MCPs = run_test(A, tau, test_set_dir, test_num, graph_name, opt_params, schedules, trackers)
        run_times.append(times)
        final_iters.append(iters)
        final_MCPs.append(MCPs)
    
    if(opt_comparison):
        plot_optimizer_comparison(test_set_dir, test_spec, run_times, final_iters, final_MCPs)


# Explore optima for the given graph and attack duration:
def run_test(A, tau, test_set_dir, test_num, graph_name, opt_params, schedules, trackers, save=True):
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    num_init_Ps = opt_params["num_init_Ps"]
    init_Ps = init_rand_Ps(A, num_init_Ps)
    learning_rates = []
    lr_scales = []
    conv_times = []
    opt_metrics = {track : [] for track in trackers}

    if(save):
        # Create directory for saving the results for the given graph and tau value:
        test_name = "test" + str(test_num) + "_" + graph_name + "_tau" + str(tau)
        test_dir = os.path.join(test_set_dir, test_name)
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        test_results_dir = os.path.join(test_dir, "results")
        if not os.path.isdir(test_results_dir):
            os.mkdir(test_results_dir)
        metrics_dir = os.path.join(test_dir, "metrics")
        if not os.path.isdir(metrics_dir):
            os.mkdir(metrics_dir)

    # Run the optimization algorithm:
    for k in range(num_init_Ps):
        print("Optimizing with initial P matrix number " + str(k + 1) + "...")
        print("Using optimizer: " + opt_params["optimizer_name"])
        P0 = init_Ps[:, :, k]
        lr, lr_scale = set_learning_rate(P0, A, F0, tau, opt_params["num_LCPs"], opt_params["nominal_learning_rate"], opt_params["grad_mode"])
        lr_scales.append(lr_scale)
        learning_rates.append(lr)
        opt_params["scaled_learning_rate"] = lr
        # --------------------------------------------------------------------
        start_time = time.time()
        P, F, tracked_vals = test_optimizer_grad_conv(P0, A, F0, tau, opt_params, schedules, trackers)
        conv_time = time.time() - start_time
        print("--- Optimization took: %s seconds ---" % (conv_time))
        # --------------------------------------------------------------------
        conv_times.append(conv_time)
        
        if(save):
            # Save initial and optimized P matrices:
            np.savetxt(test_results_dir + "/init_P_" + str(k + 1) + ".csv", P0, delimiter=',')
            np.savetxt(test_results_dir + "/opt_P_" + str(k + 1) + ".csv", P, delimiter=',')
            # Save metrics tracked during the optimization process:
            for track in trackers:
                if track.find("final") == -1:
                    opt_metrics[track].append(tracked_vals[track])
                else:
                    opt_metrics[track].extend(tracked_vals[track])
                with open(metrics_dir + "/debug.txt", 'w') as metric_debug_file:
                    metric_debug_file.write(str(opt_metrics["P_diff_sums"]))

    if(save):
        np.savetxt(test_dir + "/A.csv", np.asarray(A).astype(int), delimiter=',')  # Save the env graph binary adjacency matrix
        draw_env_graph(A, graph_name, test_dir)  # Save a drawing of the env graph
        plot_opt_metrics(opt_metrics, test_name, test_dir) # Save plots of the optimization metrics tracked
        graph_code = gen_graph_code(A)  # Generate unique graph code

        # Save all optimization metrics which were tracked:
        for metric_name in opt_metrics.keys():
            metric_path = os.path.join(metrics_dir, metric_name + ".txt")
            with open(metric_path, 'w') as metric_file:
                metric_string = ""
                for r in range(len(opt_metrics[metric_name])):
                    metric_string = metric_string + str(np.asarray(opt_metrics[metric_name][r])) + "\n"
                metric_file.write(metric_string)
                
        # Write info file with graph and optimization algorithm info:
        info_path = os.path.join(test_dir, "test" + str(test_num) + "_info.txt")
        with open(info_path, 'w') as info:
            info.write("---------- Graph Information ----------\n")
            info.write("Number of nodes (n) = " + str(n) + "\n")
            info.write("Graph Name = " + graph_name + "\n")
            info.write("Attack Duration (tau) = " + str(tau) + "\n")
            info.write("Graph Code = " + graph_code + "\n")
            info.write("\n---------- Optimizer Information ----------\n")
            info.write("Optimizer used = " + opt_params["optimizer_name"] + "\n")
            info.write("\nOptimizer Parameters from Test Specification:\n")
            del opt_params["scaled_learning_rate"]
            info.write(json.dumps(opt_params, sort_keys=False, indent=4) + "\n")
            if(opt_params["use_learning_rate_schedule"] or opt_params["use_P_update_bound_schedule"] or opt_params["use_num_LCPs_schedule"]):
                info.write("\nSchedules used:\n")
                if(opt_params["use_learning_rate_schedule"]):
                    info.write("learning_rate_schedule:\n")
                    info.write(json.dumps(schedules["learning_rate_schedule"]) + "\n")
                if(opt_params["use_P_update_bound_schedule"]):
                    info.write("P_update_elt_bound_schedule:\n")
                    info.write(json.dumps(schedules["P_update_elt_bound_schedule"]) + "\n")
                if(opt_params["use_num_LCPs_schedule"]):
                    info.write("num_LCPs_schedule:\n")
                    info.write(json.dumps(schedules["lcp_num_schedule"]) + "\n")
            info.write("\nOptimizer Parameters computed during testing:\n")
            info.write("Scaled Learning Rates = " + str(np.asarray(learning_rates)) + "\n")
            info.write("Max absolute-value elements of initial MCP gradients = " + str(np.asarray(lr_scales)) + "\n")
            # info.write("Final MCPs achieved = " + str(np.asarray(opt_metrics["final_MCP"]).flatten()) + "\n")
            info.write("Final MCPs achieved = " + str(np.asarray(opt_metrics["final_MCP"])) + "\n")
            # info.write("Number of iterations required = " + str(np.asarray(opt_metrics["final_iters"]).flatten()) + "\n")
            info.write("Number of iterations required = " + str(opt_metrics["final_iters"]) + "\n")
            info.write("Optimization Times Required (seconds) = " + str(conv_times) + "\n")
        info.close()
    return conv_times, opt_metrics["final_iters"], opt_metrics["final_MCP"]

def test_optimizer_grad_conv(P0, A, F0, tau, opt_params, schedules, trackers):
    check_time = time.time()
    n = P0.shape[0]
    P = P0 
    Q = P0
    old_MCP = 0

    optimizer = setup_optimizer(opt_params)
    opt_state = optimizer.init(P0)

    conv_test_vals = deque()  # queue storing recent values of desired metric, for checking convergence
    P_update_bound = jnp.full((n, n), opt_params["P_update_elt_bound"])
    num_LCPs = opt_params["num_LCPs"]
    tracked_vals = {track_val : [] for track_val in trackers}

    if opt_params["use_learning_rate_schedule"]:
        lr_schedule = schedules["learning_rate_schedule"]
    if opt_params["use_P_update_bound_schedule"]:
        P_update_bound_schedule = schedules["P_update_elt_bound_schedule"]
    if opt_params["use_num_LCPs_schedule"]:
        lcp_num_schedule = schedules["lcp_num_schedule"]
    
    iter = 0  # number of gradient ascent steps taken so far
    converged = False
    while not converged:
        # Apply desired scheduling:
        if opt_params["use_learning_rate_schedule"]:
            if lr_schedule["iters"].count(iter) != 0:
                index = lr_schedule["iters"].index(iter)
                new_lr = opt_params["scaled_learning_rate"]*lr_schedule["scaled_learning_rate_multipliers"][index]
                if opt_params["use_momentum"]:
                    optimizer = optax.sgd(new_lr, momentum=opt_params["mom_decay_rate"], nesterov=opt_params["use_nesterov"])
                else:
                    optimizer = optax.sgd(new_lr)
                if opt_params["grad_mode"].find("parametrization") != -1:
                    opt_state = optimizer.init(Q)
                else:
                    opt_state = optimizer.init(P)
                print("Updated scaled_learning_rate to: " + str(new_lr) + " at iteration " + str(iter))
        if opt_params["use_P_update_bound_schedule"]:
            if P_update_bound_schedule["iters"].count(iter) != 0:
                index = P_update_bound_schedule["iters"].index(iter)
                P_update_bound = jnp.full((n, n), P_update_bound_schedule["bounds"][index])
                print("Updated P_update_elt_bound to: " + str(P_update_bound_schedule["bounds"][index]) + " at iteration " + str(iter))
        if opt_params["use_num_LCPs_schedule"]:
            if lcp_num_schedule["iters"].count(iter) != 0:
                index = lcp_num_schedule["iters"].index(iter)
                num_LCPs = lcp_num_schedule["lcp_nums"][index]
                print("Updated P_update_elt_bound to: " + str(num_LCPs) + " at iteration " + str(iter))

        # compute gradient:
        grad = get_grad(A, P, Q, F0, tau, num_LCPs, opt_params)

        # get update to the P matrix or Q matrix:
        updates, opt_state = optimizer.update(grad, opt_state)
        # bound the update to the P matrix if using projection-based approach: [TEST TO SEE IF Q BOUND NEEDED]
        if opt_params["grad_mode"].find("projection") != -1:
            updates = jnp.minimum(updates, P_update_bound)
            updates = jnp.maximum(updates, -1*P_update_bound)

        # apply update to P matrix and compute P_diff matrix:
        if opt_params["grad_mode"].find("parametrization") != -1:
            P_old = comp_P_param(Q, A)
            Q = optax.apply_updates(Q, updates)
            P = comp_P_param(Q, A)
        else:
            P_old = P
            P = optax.apply_updates(P, updates)
            P = proj_onto_simplex(P)
    
        P_diff = P - P_old
        abs_P_diff_sum = jnp.sum(jnp.abs(P_diff))

        # track metrics of interest:
        if iter % opt_params["iters_per_trackvals"] == 0:
            tracked_vals["iters"].append(iter)
            tracked_vals["P_diff_sums"].append(abs_P_diff_sum)
            tracked_vals["P_diff_max_elts"].append(jnp.max(jnp.abs(P_diff)))
            F = compute_cap_probs(P, F0, tau)
            F = F.reshape((n**2), order='F')
            tracked_vals["MCP_inds"].append(jnp.argmin(F))
            tracked_vals["MCPs"].append(jnp.min(F))
            # print status update to terminal:
            if(iter % opt_params["iters_per_printout"] == 0):
                print("------ iteration number " + str(iter) + ", elapsed time =  " + str(time.time() - check_time) + "-------")
                print("grad 1-norm = " + str(jnp.sum(jnp.abs(grad))))
                print("grad inf norm = " + str(jnp.max(jnp.abs(grad))))
                print("abs_P_diff_sum = " + str(jnp.sum(jnp.abs(P_diff))))
                print("MCP = " + str(jnp.min(F)))

        if opt_params["conv_test_mode"] == "P_update":
            converged, conv_test_vals = conv_check(iter, abs_P_diff_sum, conv_test_vals, opt_params)
        elif opt_params["conv_test_mode"] == "MCP_diff":
            MCP =  compute_MCP(P, F0, tau)
            MCP_diff = MCP - old_MCP
            converged, conv_test_vals = conv_check(iter, MCP_diff, conv_test_vals, opt_params)
            old_MCP = MCP
        if iter == opt_params["max_iters"]:
            converged = True
        iter = iter + 1

    # convergence or max iteration count reached...
    F = compute_cap_probs(P, F0, tau)
    final_MCP = jnp.min(F)
    tracked_vals["final_MCP"].append(final_MCP)
    print("Minimum Capture Probability at iteration " + str(iter) + ":")
    print(final_MCP)
    tracked_vals["final_iters"].append(iter)
    return P, F, tracked_vals

def setup_optimizer(opt_params):
    if opt_params["optimizer_name"] == "sgd":
        if opt_params["use_momentum"]:
            optimizer = optax.sgd(opt_params["scaled_learning_rate"], momentum=opt_params["mom_decay_rate"], nesterov=opt_params["use_nesterov"])
        else:
            optimizer = optax.sgd(opt_params["scaled_learning_rate"])
    elif opt_params["optimizer_name"] == "adagrad":
        optimizer = optax.adagrad(opt_params["scaled_learning_rate"])
    elif opt_params["optimizer_name"] == "adam":
        optimizer = optax.adam(opt_params["scaled_learning_rate"])
    elif opt_params["optimizer_name"] == "rmsprop":
        if opt_params["use_momentum"]:
            optimizer = optax.rmsprop(opt_params["scaled_learning_rate"], momentum=opt_params["mom_decay_rate"], nesterov=opt_params["use_nesterov"])
        else:
            optimizer = optax.rmsprop(opt_params["scaled_learning_rate"])
    return optimizer

def get_grad(A, P, Q, F0, tau, num_LCPs, opt_params):
        if opt_params["grad_mode"] == "MCP_parametrization":
            grad = -1*comp_MCP_grad_param(Q, A, F0, tau) # negate so that the optimizer does gradient ascent
        elif opt_params["grad_mode"] == "MCP_abs_parametrization":
            grad = -1*comp_MCP_grad_param_abs(Q, A, F0, tau) # negate so that the optimizer does gradient ascent
        elif opt_params["grad_mode"] == "LCP_parametrization":
            grad = -1*comp_avg_LCP_grad_param(Q, A, F0, tau, num_LCPs) # negate so that the optimizer does gradient ascent
        elif opt_params["grad_mode"] == "MCP_projection":
            grad = -1*comp_MCP_grad(P, F0, tau) # negate so that the optimizer does gradient ascent
        elif opt_params["grad_mode"] == "LCP_projection":
            grad = -1*comp_avg_LCP_grad(P, F0, tau, num_LCPs) # negate so that the optimizer does gradient ascent
        else:
            raise ValueError("Invalid grad_mode specified!")
        return grad

# Check for convergence in moving average of chosen convergence metric:
def conv_check(iter, new_val, conv_test_vals, opt_params):
    conv_test_vals.append(new_val)
    if iter > opt_params["conv_window_size"]:
        conv_test_vals.popleft()
    MA_val = np.mean(conv_test_vals)
    if iter > opt_params["conv_window_size"]:
        converged = MA_val < opt_params["conv_radius"]
    else:
        converged = False
    if iter == opt_params["max_iters"]:
        converged = True
    return converged, conv_test_vals

# TESTING ------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    test_set_name = "Complete_Graph_Testing4"
    test_spec = ts.TestSpec(test_spec_filepath=os.getcwd() + "/TestSpecs/complete_graph_test_spec_v2.json")

    test_start_time = time.time()
    run_test_set(test_set_name, test_spec)
    print("Running test_set_" + test_set_name + " took " + str(time.time() - test_start_time) + " seconds to complete.")

    # A, graph_name = gen_grid_G(3, 3)
    # tau = graph_diam(A)
    # num_init_Ps = 5
    # max_iters = 1000
    # print("Graph name: " + graph_name)
    # test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters)

    # test_start_time = time.time()
    # for i in range(1):
    #     for j in range(1):
    #         for k in range(len(test_graphs)):
    #             print("-------- Working on Graph Number " + str(graph_num) + "----------")
    #             A = test_graphs[k]
    #             tau = test_taus[k]
    #             explore_graph_optima(A, tau, test_set_name, graph_num, num_init_Ps, grad_mode="MCP_parametrization")
    #             graph_num = graph_num + 1
    # print("Running test_set_" + test_set_name + " took " + str(time.time() - test_start_time) + " seconds to complete.")

    # Visualization: [NEED TO FIX CIRCULAR IMPORT WITH visualize_results for avg_opt_P_mat drawing]
    # visualize_metrics(test_set_name, overlay=True)
    # visualize_results(test_set_name)






    