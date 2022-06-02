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

# Run desired optimizer for a fixed number of iterations
def test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters, grad_mode):
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
    grad_mode : String
        The type of gradient computation to perform.
    
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
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    print("comp_MCP_grad_param id:")
    print(id(comp_MCP_grad_param))
    # print("comp_MCP_grad_param_extra id:")
    # print(id(comp_MCP_grad_param_extra))
    # print("comp_MCP_grad_param_test id:")
    # print(id(comp_MCP_grad_param_test))
    time_avgs = []
    for k in range(num_init_Ps):
        # print("---------------------Optimizing with initial P matrix number " + str(k + 1) + "----------------------------")
        P0 = init_Ps[:, :, k]
        P = P0
        Q = P0
        # print("P0 shape = " + str(jnp.shape(P0)) + ", F0 shape = " + str(jnp.shape(F0)) + ", A shape = " + str(jnp.shape(A)))
        scaled_learning_rate, _ = set_learning_rate(P0, A, F0, tau, num_LCPs, nominal_learning_rate, grad_mode)
        # print("scaled_learning_rate = " + str(scaled_learning_rate))
        # optimizer = optax.rmsprop(scaled_learning_rate)
        # optimizer = optax.rmsprop(scaled_learning_rate, momentum=0.9, nesterov=False)
        optimizer = optax.sgd(scaled_learning_rate, momentum=0.99, nesterov=True)
        opt_state = optimizer.init(P0)
        check_time = time.time()
        for iter in range(max_iters + 1):
            # print("iteration number: " + str(k))
            # start_time = time.time()
            if grad_mode == "MCP_parametrization":
                grad = -1*comp_MCP_grad_param(Q, A, F0, tau) # negate so that the optimizer does gradient ascent
            # elif grad_mode == "MCP_extra_parametrization":
            #     grad = -1*comp_MCP_grad_param_extra(Q, A, F0, tau, num_LCPs) # negate so that the optimizer does gradient ascent
            # elif grad_mode == "MCP_test_parametrization":
            #     grad = -1*comp_MCP_grad_param_test(Q, A, F0, tau) # negate so that the optimizer does gradient ascent
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
            # if iter % 200 == 0:
            # # if iter % 1 == 0: 
            #     print("------ iteration number " + str(iter) + ", elapsed time =  " + str(time.time() - check_time)[:7] + "-------")
            #     print("grad 1-norm = " + str(jnp.sum(jnp.abs(grad))))
            #     print("grad inf norm = " + str(jnp.max(jnp.abs(grad))))
            #     print("MCP = " + str(compute_MCP(P, F0, tau)))
        opt_time = time.time() - check_time
        time_avgs.append(opt_time)
        print("Optimizing P matrix number " + str(k + 1) + " over " + str(max_iters) + " iterations took: " + str(opt_time)[:7] + " seconds")
        # print("Final MCP:")
        # print(compute_MCP(P, F0, tau))
    return time_avgs

# Run desired optimizer for a fixed number of iterations 
def test_optimizer_fixed_iters_test(A, tau, num_init_Ps, max_iters):
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
    num_LCPs=1
    nominal_learning_rate = 0.0001
    grad_mode = "MCP_parametrization"
    grad_func = get_grad_func(grad_mode)
    print("grad_func id:")
    print(id(grad_func))
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    time_avgs = []
    for k in range(num_init_Ps):
        # print("---------------------Optimizing with initial P matrix number " + str(k + 1) + "----------------------------")
        P0 = init_Ps[:, :, k]
        P = P0
        Q = P0
        # print("P0 shape = " + str(jnp.shape(P0)) + ", F0 shape = " + str(jnp.shape(F0)) + ", A shape = " + str(jnp.shape(A)))
        scaled_learning_rate, _ = set_learning_rate(P0, A, F0, tau, num_LCPs, nominal_learning_rate, grad_mode)
        # print("scaled_learning_rate = " + str(scaled_learning_rate))
        # optimizer = optax.rmsprop(scaled_learning_rate)
        # optimizer = optax.rmsprop(scaled_learning_rate, momentum=0.9, nesterov=False)
        optimizer = optax.sgd(scaled_learning_rate, momentum=0.99, nesterov=True)
        opt_state = optimizer.init(P0)
        check_time = time.time()
        for iter in range(max_iters + 1):
            # start_time = time.time()
            grad = -1*grad_func(Q, A, F0, tau)
            # grad = grad_comp_test(P, Q, A, F0, tau, num_LCPs, grad_mode)
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
            # if iter % 200 == 0:
            # # if iter % 1 == 0: 
            #     print("------ iteration number " + str(iter) + ", elapsed time =  " + str(time.time() - check_time)[:7] + "-------")
            #     print("grad 1-norm = " + str(jnp.sum(jnp.abs(grad))))
            #     print("grad inf norm = " + str(jnp.max(jnp.abs(grad))))
            #     print("MCP = " + str(compute_MCP(P, F0, tau)))
        opt_time = time.time() - check_time
        time_avgs.append(opt_time)
        print("Optimizing P matrix number " + str(k + 1) + " over " + str(max_iters) + " iterations took: " + str(opt_time)[:7] + " seconds")
        # print("Final MCP:")
        # print(compute_MCP(P, F0, tau))
    return time_avgs

def set_learning_rate(P0, A, F0, tau, num_LCPs, nominal_LR, grad_mode):
    if grad_mode == "MCP_projection":
        init_grad = comp_MCP_grad(P0, A, F0, tau)
    elif grad_mode == "MCP_parametrization":
        init_grad = comp_MCP_grad_param(P0, A, F0, tau)
    # elif grad_mode == "MCP_test_parametrization":
    #     init_grad = comp_MCP_grad_param_test(P0, A, F0, tau)
    # elif grad_mode == "MCP_extra_parametrization":
    #     init_grad = comp_MCP_grad_param_extra(P0, A, F0, tau, num_LCPs)
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
        test_start_time = time.time()
        print("-------- Working on Graph " + graph_name + " with tau = " + str(tau) + "----------")
        times, iters, MCPs = run_test(A, tau, test_set_dir, test_num, graph_name, opt_params, schedules, trackers)
        print("Running test number " + str(test_num) + " took " + str(time.time() - test_start_time) + " seconds to complete.")
        
        run_times.append(times)
        final_iters.append(iters)
        final_MCPs.append(MCPs)
    
    if(opt_comparison):
        plot_optimizer_comparison(test_set_dir, test_spec, run_times, final_iters, final_MCPs)

# Explore optima for the given graph and attack duration:
def run_test(A, tau, test_set_dir, test_num, graph_name, opt_params, schedules, trackers, save=True):
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
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    num_init_Ps = opt_params["num_init_Ps"]
    init_Ps = init_rand_Ps(A, num_init_Ps)
    learning_rates = []
    lr_scales = []
    conv_times = []
    opt_metrics = {track : [] for track in trackers}
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
        P, F, tracked_vals = run_optimizer(P0, A, F0, tau, opt_params, schedules, trackers)
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

    if(save):
        np.savetxt(test_dir + "/A.csv", np.asarray(A).astype(int), delimiter=',')  # Save the env graph binary adjacency matrix
        draw_env_graph(A, graph_name, test_dir)  # Save a drawing of the env graph
        visualize_metrics(opt_metrics, test_name, test_dir, show_legend=False) # Save plots of the optimization metrics tracked
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

def run_optimizer(P0, A, F0, tau, opt_params, schedules, trackers):
    check_time = time.time()
    n = P0.shape[0]
    P = P0 
    Q = P0
    old_MCP = 0

    optimizer = setup_optimizer(opt_params)
    opt_state = optimizer.init(P0)
    grad_func = get_grad_func(opt_params["grad_mode"])

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
        if opt_params["use_learning_rate_schedule"]:  #currently only implemented for SGD
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


        # apply update to P matrix, and parametrization Q, if applicable:
        if opt_params["grad_mode"].find("parametrization") != -1:
            grad = -1*grad_func(Q, A, F0, tau) # compute negative gradient
            P_old = comp_P_param(Q, A)
            # get update to the parametrization Q:
            updates, opt_state = optimizer.update(grad, opt_state)
            Q = optax.apply_updates(Q, updates)
            P = comp_P_param(Q, A) # compute new P matrix
        else:
            grad = -1*grad_func(P, A, F0, tau) # compute negative gradient
            P_old = P
            # get update to the P matrix:
            updates, opt_state = optimizer.update(grad, opt_state)
            updates = jnp.minimum(updates, P_update_bound)
            updates = jnp.maximum(updates, -1*P_update_bound)
            P = optax.apply_updates(P, updates) 
            P = proj_onto_simplex(P) # project rows onto simplexes to get valid new P matrix
    
        # Compute the difference between the latest P matrix and the previous one:
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

# Simulated annealing approach to finding optimal strategy:
def sim_anneal(Q0, A, F0, tau, elt_step_size, init_temp, max_iters, iters_per_print):
    check_time = time.time()
    n = A.shape[0]
    Q = Q0
    best_Q = Q
    P = comp_P_param(Q, A)
    best_MCP = compute_MCP(P, F0, tau)
    curr_Q, curr_MCP = best_Q, best_MCP
    key = jax.random.PRNGKey(1)
    iter = 0  # number of gradient ascent steps taken so far
    for iter in range(max_iters + 1):
        # if(iter % iters_per_print == 0):
        #     print("------ iteration number " + str(iter) + ", elapsed time =  " + str(time.time() - check_time) + "-------")
        #     print("best_MCP = " + str(best_MCP))
        #     print("curr_MCP = " + str(curr_MCP))
        key, subkey = jax.random.split(key)
        candidate_Q = curr_Q + elt_step_size*jax.random.uniform(subkey, (n, n))
        candidate_P = comp_P_param(candidate_Q, A)
        candidate_MCP = compute_MCP(candidate_P,  F0, tau)
        if candidate_MCP > best_MCP:
            best_Q, best_MCP = candidate_Q, candidate_MCP
        MCP_diff = curr_MCP - candidate_MCP
        temp = init_temp/float(iter + 1)
        metropolis = jnp.exp(-MCP_diff/temp)
        key, subkey = jax.random.split(key)
        if MCP_diff > 0 or jax.random.uniform(subkey) < metropolis:
            curr_Q, curr_MCP = candidate_Q, candidate_MCP
    best_P = comp_P_param(best_Q, A)
    return best_Q, best_P, best_MCP

def sim_anneal_test(Q0, A, F0, tau, elt_step_size, init_temp, max_iters, iters_per_print):
    check_time = time.time()
    n = A.shape[0]
    Q = Q0
    best_Q = Q
    P = comp_P_param_abs(Q, A)
    best_MCP = compute_MCP(P, F0, tau)
    curr_Q, curr_MCP = best_Q, best_MCP
    print("initial MCP = ")
    print(curr_MCP)
    key = jax.random.PRNGKey(1)
    iter = 0  # number of gradient ascent steps taken so far
    for iter in range(max_iters + 1):
        # if(iter % iters_per_print == 0):
        #     print("------ iteration number " + str(iter) + ", elapsed time =  " + str(time.time() - check_time) + "-------")
        #     print("best_MCP = " + str(best_MCP))
        #     print("curr_MCP = " + str(curr_MCP))
        curr_Q, curr_MCP, best_Q, best_MCP = sim_anneal_inner(curr_Q, curr_MCP, best_Q, best_MCP, iter, A, F0, tau, elt_step_size, init_temp, key)
        time.sleep(0.005)
    best_P = comp_P_param_abs(best_Q, A)
    return best_Q, best_P, best_MCP

@functools.partial(jit, static_argnames=['tau'])
def sim_anneal_inner(curr_Q, curr_MCP, best_Q, best_MCP, iter, A, F0, tau, elt_step_size, init_temp, key):
    key, subkey = jax.random.split(key)
    candidate_Q = curr_Q + elt_step_size*jax.random.uniform(subkey, (n, n), minval=-1.0, maxval=1.0)
    candidate_P = comp_P_param_abs(candidate_Q, A)
    candidate_MCP = compute_MCP(candidate_P,  F0, tau)
    # if candidate_MCP > best_MCP:
    MCP_array = jnp.full(2, np.NaN)
    MCP_array = MCP_array.at[0].set(candidate_MCP)
    MCP_array = MCP_array.at[1].set(best_MCP)
    Q_array = jnp.full((n, n, 2), np.NaN)
    Q_array = Q_array.at[:, :, 0].set(candidate_Q)
    Q_array = Q_array.at[:, :, 1].set(best_Q)
    best_MCP = jnp.max(MCP_array)
    best_ind = jnp.argmax(MCP_array)
    best_Q = Q_array[:, :, best_ind]
    MCP_diff = curr_MCP - candidate_MCP
    temp = init_temp/jnp.array(iter + 1, dtype=float)
    metropolis = jnp.exp(-MCP_diff/temp)
    key, subkey = jax.random.split(key)
    # reset arrays to hold candidate and curr:
    Q_array = Q_array.at[:, :, 0].set(candidate_Q)
    Q_array = Q_array.at[:, :, 1].set(curr_Q)
    MCP_array = MCP_array.at[0].set(candidate_MCP)
    MCP_array = MCP_array.at[1].set(curr_MCP)
    # check if MCP_diff is > 0:
    MCP_diff_array = jnp.full(2, np.NaN)
    MCP_diff_array = MCP_diff_array.at[0].set(MCP_diff)
    MCP_diff_array = MCP_diff_array.at[1].set(0)
    zero_check_ind = jnp.argmax(MCP_diff_array)
    # update curr_Q and curr_MCP if so:
    curr_Q = Q_array[:, :, zero_check_ind]
    curr_MCP = MCP_array[zero_check_ind]
    # check if metropolis criterion satisfied and update curr_Q and curr_MCP if so:
    metropolis_array = jnp.full(2, np.NaN)
    metropolis_array = metropolis_array.at[0].set(jax.random.uniform(subkey))
    metropolis_array = metropolis_array.at[1].set(metropolis)
    metropolis_ind = jnp.argmin(metropolis_array)
    curr_Q = Q_array[:, :, metropolis_ind]
    curr_MCP = MCP_array[metropolis_ind]
    return curr_Q, curr_MCP, best_Q, best_MCP

# TESTING ------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    test_set_name = "Split_Star_Study2"
    test_spec = ts.TestSpec(test_spec_filepath=os.getcwd() + "/TestSpecs/splitstar_study_v2.json")

    test_set_start_time = time.time()
    run_test_set(test_set_name, test_spec)
    print("Running test_set_" + test_set_name + " took " + str(time.time() - test_set_start_time) + " seconds to complete.")




    # res_vis_dir = os.getcwd() + "/Results/test_set_InitP1000_Study_3x3Grid1/test1_grid_W3_H3_tau4/results_visualization"
    # best_opt_P = np.loadtxt(res_vis_dir + "/sym_transformed_opt_P_887.csv", delimiter=',')

    # A, graph_name = gen_grid_G(3, 3)
    # n = A.shape[0]
    # tau = 4
    # F0 = jnp.full((n, n, tau), np.NaN)
    # test_opt_P = np.array([[0, 0.467, 0, 0.533, 0, 0, 0, 0, 0],
    #                        [0.3, 0, 0, 0, 0.7, 0, 0, 0, 0],
    #                        [0, 1.00, 0, 0, 0, 0, 0, 0, 0], 
    #                        [0.337, 0, 0, 0, 0.575, 0, 0.088, 0, 0],
    #                        [0, 0, 0, 0, 0, 0.60, 0, 0.40, 0],
    #                        [0, 0, 0.567, 0, 0, 0, 0, 0, 0.433],
    #                        [0, 0, 0, 1.00, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0.4433, 0, 0.52, 0, 0.0367],
    #                        [0, 0, 0, 0, 0, 0.30, 0, 0.70, 0]])                       
    # test_opt_P = jnp.asarray(test_opt_P, dtype=float)
    # MCP = compute_MCP(test_opt_P, F0, tau)
    # print("test_opt_P = ")
    # print(np.asarray(test_opt_P))    
    # print("MCP = ")
    # print(MCP)
    # print("best_opt_P = ")
    # print(best_opt_P)
    # best_opt_P = jnp.asarray(best_opt_P)
    # MCP = compute_MCP(best_opt_P, F0, tau)
    # print("MCP = ")
    # print(MCP)



    # visualize_results(test_set_name)
    # visualize_MCPs(test_set_name, tau_study=True, num_top_MCPs=None, plot_best_fit=True)
    # visualize_MCPs(test_set_name, tau_study=True, num_top_MCPs=1, plot_best_fit=True)
    # visualize_MCPs(test_set_name, tau_study=True, num_top_MCPs=5, plot_best_fit=True)
    # visualize_MCPs(test_set_name, tau_study=True, num_top_MCPs=10, plot_best_fit=True)

    # A, graph_name = gen_grid_G(3, 3)
    # num_init_Ps = 1
    # init_Ps = init_rand_Ps(A, num_init_Ps)
    # tau = graph_diam(A)
    # P0 = init_Ps[:, :, 0]
    # n = A.shape[0]
    # F0 = jnp.full((n, n, tau), np.NaN)
    # P0 = init_Ps[:, :, 1]
    # elt_step_size = 0.1
    # init_temp = 500
    # max_iters = 5000
    # iters_per_print = 100
    # print("Graph Name: " + graph_name)
    # print("tau = " + str(tau))
    # # print("init_P_num = " + str(i))
    # best_Q, best_P, best_MCP = sim_anneal_test(P0, A, F0, tau, elt_step_size, init_temp, max_iters, iters_per_print)
    # # max_iters = 10000
    # init_temp = 100
    # for i in range(50):
    #     print("Current best MCP: " + str(best_MCP))
    #     elt_step_size = 0.1/(i + 1)
    #     best_Q, best_P, best_MCP = sim_anneal_test(best_Q, A, F0, tau, elt_step_size, init_temp, max_iters, iters_per_print)

    # print(best_MCP)
    # print(best_Q)
    # print(best_P)
    
    # A, graph_name = gen_grid_G(3, 5)
    # tau = graph_diam(A)
    # num_init_Ps = 15
    # max_iters = 2000
    # print("Graph name: " + graph_name)    

    # print("--------------------- Speed testing standard jacrev...")
    # test_time_avgs = test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters, "MCP_parametrization")
    # print("Optimizing took " + str(np.mean(np.asarray(test_time_avgs))) + " seconds to complete on avg")

    # print("---------------------Speed testing with grad...")
    # time_avgs = test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters, "MCP_test_parametrization")
    # print("Optimizing (grad) took " + str(np.mean(np.asarray(time_avgs))) + " seconds to complete on avg")

    # print("---------------------Speed testing with extra param...")
    # time_avgs = test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters, "MCP_extra_parametrization")
    # print("Optimizing (extra param) took " + str(np.mean(np.asarray(time_avgs))) + " seconds to complete on avg")

    # print("--------------------- Speed testing standard jacrev...")
    # test_time_avgs = test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters, "MCP_parametrization")
    # print("Optimizing took " + str(np.mean(np.asarray(test_time_avgs))) + " seconds to complete on avg")

    # print("---------------------Speed testing with grad...")
    # time_avgs = test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters, "MCP_test_parametrization")
    # print("Optimizing (grad) took " + str(np.mean(np.asarray(time_avgs))) + " seconds to complete on avg")

    # print("---------------------Speed testing with extra param...")
    # time_avgs = test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters, "MCP_extra_parametrization")
    # print("Optimizing (extra param) took " + str(np.mean(np.asarray(time_avgs))) + " seconds to complete on avg")

    
    # Visualization: [NEED TO FIX CIRCULAR IMPORT WITH visualize_results for avg_opt_P_mat drawing]
    # visualize_metrics(test_set_name, overlay=True)
    # visualize_results(test_set_name)






    