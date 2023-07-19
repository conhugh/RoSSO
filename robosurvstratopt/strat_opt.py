# Optimization of the performance of stochastic surveillance strategies
from collections import deque
import json
import os
import shutil
import time

import jax.numpy as jnp
import numpy as np
import optax

import graph_comp
import strat_comp
import strat_viz
from test_spec import TestSpec


def test_optimizer_fixed_iters(A, tau, num_init_Ps, max_iters):
    """
    Test and time various Optax optimizers for a fixed number of iterations.

    Parameters
    ----------
    A : 
    tau : int
        Intruder's attack duration. 
    num_init_Ps : int
        Number of random initializations.
    max_iters : int
        Total number of iterations.
    
    Returns
    -------
    numpy.ndarray
        Transition probability matrix corresponding to optimized strategy.
    numpy.ndarray
        Capture Probability matrix corresponding to optimized strategy.
    """
    init_Ps = strat_comp.init_rand_Ps(A, num_init_Ps)
    num_LCPs = 1
    nominal_learning_rate = 0.001
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    time_avgs = []
    for k in range(num_init_Ps):
        Q = init_Ps[:, :, k]
        init_grad = strat_comp.comp_avg_LCP_grad(Q, A, F0, tau, num_LCPs)
        init_grad_max = jnp.max(jnp.abs(init_grad))
        scaled_learning_rate = nominal_learning_rate/init_grad_max
        schedule = optax.constant_schedule(scaled_learning_rate)
        # optimizer = optax.rmsprop(schedule)
        # optimizer = optax.rmsprop(schedule, momentum=0.9, nesterov=False)
        optimizer = optax.sgd(schedule, momentum=0.99, nesterov=True)
        opt_state = optimizer.init(Q)
        check_time = time.time()
        for iter in range(max_iters + 1):
            # start_time = time.time()
            grad = -1*grad_func(Q, A, F0, tau, num_LCPs)
            # print("--- Getting grad took: %s seconds ---" % (time.time() - start_time))
            # start_time = time.time()
            updates, opt_state = optimizer.update(grad, opt_state)
            # print("--- Getting update and state took: %s seconds ---" % (time.time() - start_time))
            # start_time = time.time()
            Q = optax.apply_updates(Q, updates)
            # print("--- Applying update took: %s seconds ---" % (time.time() - start_time))
        P = strat_comp.comp_P_param(Q, A)
        print(P)
        opt_time = time.time() - check_time
        time_avgs.append(opt_time)
        print("Optimizing P matrix number " + str(k + 1) + " over " + str(max_iters) + " iterations took: " + str(opt_time)[:7] + " seconds")
        print("Final MCP: " + str(strat_comp.compute_LCPs(P, F0, tau, num_LCPs)))
    return time_avgs

def run_test_set(test_set_name, test_spec=None, opt_comparison=False):
    """
    Run computational study with graphs, attack durations, and optimization parameters defined in test spec.

    Parameters
    ----------
    test_set_name : String
        Folder name to use for saving results from computational study. 
    test_spec : TestSpec
        Test specification defining parameters for the computational study.
    opt_comparison : bool
        Flag indicating whether multiple optimizer types are being compared. 

    See Also
    -------
    test_spec.TestSpec
    run_test
    """
    if type(test_spec) != TestSpec:
        raise ValueError("Must provide a TestSpec object for test specification.")
    test_spec.validate_test_spec()

    # create directory for saving results:
    test_set_dir = "./results/local/test_set_" + str(test_set_name)
    if os.path.isdir(test_set_dir):
        input("WARNING! Results from a test with the same name have been saved previously, press ENTER to overwrite existing results.")
        for files in os.listdir(test_set_dir):
            path = os.path.join(test_set_dir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)
    else:
        os.mkdir(test_set_dir)
    # save a copy of test spec to results folder, for record-keeping:
    test_spec.save_test_spec("test_spec", test_set_dir) 

    # run all tests defined in test spec:
    run_times = []
    final_iters = []
    final_MCPs = []
    for test_num in range(1, test_spec.num_tests + 1):
        test_name = "test" + str(test_num)
        graph_name = test_spec.graph_names[test_name]
        graph_code = test_spec.graph_codes[test_name]
        A = graph_comp.graph_decode(graph_code)
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
        test_start_time = time.time()
        times, iters, MCPs = run_test(A, tau, test_set_dir, test_num, graph_name, opt_params, schedules, trackers)
        print("Running test number " + str(test_num) + " took " + str(time.time() - test_start_time) + " seconds to complete.")
        run_times.append(times)
        final_iters.append(iters)
        final_MCPs.append(MCPs)

    # generate performance comparison for optimizers used in test set:
    if(opt_comparison):
        strat_viz.plot_optimizer_comparison(test_set_dir, test_spec, run_times, final_iters, final_MCPs)

def run_test(A, tau, test_set_dir, test_num, graph_name, opt_params, schedules, trackers):
    """
    Run optimizer for the given graph, attack duration, and number of initial strategies.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph. 
    tau : int
        Intruder's attack duration.
    test_set_dir : String
        Path to directory for saving optimization results. 
    test_num : int
        Number of the current test within the test set. 
    graph_name : String
        Short name indicating structure and size of environment graph.
    opt_params : dict
        Parameters defining the optimization processes to run. 
    schedules : dict
        Schedules for applicable optimization parameters. 
    trackers : List[String]
        Quantities to track throughout optimization processes. 
    
    Returns
    ----------
    List[float]
        Times required for each optimization process to terminate. 
    List[int]
        Number of iterations required for each optimization process to terminate. 
    List[float]
        Minimum capture probabilities achieved by each optimized strategy. 
    
    See Also
    -------
    test_spec.TestSpec
    run_optimizer
    """
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
    
    # Run the optimization algorithm for the specified number of initial strategies:
    n = A.shape[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    num_init_Ps = opt_params["num_init_Ps"]
    init_Ps = strat_comp.init_rand_Ps(A, num_init_Ps)
    learning_rates = []
    lr_scales = []
    cnvg_times = []
    opt_metrics = {track : [] for track in trackers}
    for k in range(num_init_Ps):
        print("Optimizing with initial P matrix number " + str(k + 1) + "...")
        print("Using optimizer: " + opt_params["optimizer_name"])
        P0 = init_Ps[:, :, k]
        init_grad = strat_comp.comp_avg_LCP_grad(P0, A, F0, tau, opt_params["num_LCPs"], opt_params["use_abs_param"])
        lr_scale = jnp.max(jnp.abs(init_grad))
        lr = opt_params["nominal_learning_rate"]/lr_scale
        lr_scales.append(lr_scale)
        learning_rates.append(lr)
        opt_params["scaled_learning_rate"] = lr

        start_time = time.time()
        P, F, tracked_vals = run_optimizer(P0, A, F0, tau, opt_params, schedules, trackers)
        cnvg_time = time.time() - start_time
        cnvg_times.append(cnvg_time)
        print("--- Optimization took: %s seconds ---" % (cnvg_time))

        # Save initial and optimized P matrices:
        np.savetxt(test_results_dir + "/init_P_" + str(k + 1) + ".csv", P0, delimiter=',')
        np.savetxt(test_results_dir + "/opt_P_" + str(k + 1) + ".csv", P, delimiter=',')

        # Save metrics tracked during the optimization process:
        for track in trackers:
            if track.find("final") == -1:
                opt_metrics[track].append(tracked_vals[track])
            else:
                opt_metrics[track].extend(tracked_vals[track])

    # Save results from optimization processes:
    np.savetxt(test_dir + "/A.csv", np.asarray(A).astype(int), delimiter=',')  # Save the env graph binary adjacency matrix
    strat_viz.draw_env_graph(A, graph_name, test_dir)  # Save a drawing of the env graph
    strat_viz.visualize_metrics(opt_metrics, test_name, test_dir, show_legend=False) # Save plots of the optimization metrics tracked
    graph_code = graph_comp.gen_graph_code(A)  # Generate unique graph code

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
        if(opt_params["use_num_LCPs_schedule"]):
            info.write("\nSchedules used:\n")
            info.write("num_LCPs_schedule:\n")
            info.write(json.dumps(schedules["lcp_num_schedule"]) + "\n")
        info.write("\nOptimizer Parameters computed during testing:\n")
        info.write("Scaled Learning Rates = " + str(np.asarray(learning_rates)) + "\n")
        info.write("Max absolute-value elements of initial MCP gradients = " + str(np.asarray(lr_scales)) + "\n")
        info.write("Final MCPs achieved = " + str(np.asarray(opt_metrics["final_MCP"])) + "\n")
        info.write("Number of iterations required = " + str(opt_metrics["final_iters"]) + "\n")
        info.write("Optimization Times Required (seconds) = " + str(cnvg_times) + "\n")
    info.close()
    return cnvg_times, opt_metrics["final_iters"], opt_metrics["final_MCP"]

def run_optimizer(P0, A, F0, tau, opt_params, schedules, trackers):
    """
    Run optimizer for the given graph, attack duration, and initial strategy.

    Parameters
    ----------
    P0 : jaxlib.xla_extension.DeviceArray
        Initial transition probability matrix (initial strategy).
    A : jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph. 
    F0 : numpy.ndarray
        Placeholder for storing FHT probability matrices. 
    tau : int
        Intruder's attack duration.
    opt_params : dict
        Parameters defining the optimization processes to run. 
    schedules : dict
        Schedules for applicable optimization parameters. 
    trackers : List[String]
        Quantities to track throughout optimization process. 
    
    Returns
    ----------
    jaxlib.xla_extension.DeviceArray
        Optimized transition probability matrix (optimized strategy). 
    jaxlib.xla_extension.DeviceArray
        Capture probability matrix corresponding to optimized strategy.  
    dict
        Quantities tracked throughout optimization process. 
    
    See Also
    -------
    test_spec
    """
    check_time = time.time()
    n = P0.shape[0]
    P = P0 
    Q = P0
    old_MCP = 0
    diam_pairs = graph_comp.get_diametric_pairs(A)
    optimizer = setup_optimizer(opt_params)
    opt_state = optimizer.init(P0)
    grad_func = strat_comp.comp_avg_LCP_grad
    cnvg_test_vals = deque()  # queue storing recent values of desired metric, for checking convergence
    num_LCPs = opt_params["num_LCPs"]
    tracked_vals = {track_val : [] for track_val in trackers}
    # initialize parameter schedules, if specified:
    if opt_params["use_num_LCPs_schedule"]:
        lcp_num_schedule = schedules["lcp_num_schedule"]

    # Run gradient-based optimization process:
    iter = 0 
    converged = False
    while not converged:
        if opt_params["use_num_LCPs_schedule"]:
            if lcp_num_schedule["iters"].count(iter) != 0:
                index = lcp_num_schedule["iters"].index(iter)
                num_LCPs = lcp_num_schedule["lcp_nums"][index]
                print("Updated P_update_elt_bound to: " + str(num_LCPs) + " at iteration " + str(iter))
        # apply update to P matrix, and parametrization Q:
        grad = -1*grad_func(Q, A, F0, tau) # compute negative gradient
        P_old = strat_comp.comp_P_param(Q, A)
        updates, opt_state = optimizer.update(grad, opt_state)
        Q = optax.apply_updates(Q, updates)
        P = strat_comp.comp_P_param(Q, A) # compute new P matrix
        # Compute the difference between the latest P matrix and the previous one:
        P_diff = P - P_old
        abs_P_diff_sum = jnp.sum(jnp.abs(P_diff))
        # track metrics of interest:
        if iter % opt_params["iters_per_trackvals"] == 0:
            tracked_vals["iters"].append(iter)
            tracked_vals["P_diff_sums"].append(abs_P_diff_sum)
            tracked_vals["P_diff_max_elts"].append(jnp.max(jnp.abs(P_diff)))
            F = strat_comp.compute_cap_probs(P, F0, tau)
            tracked_vals["diam_pair_CP_variance"].append(strat_comp.compute_diam_pair_CP_variance(F, diam_pairs))
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
        # check for convergence:
        if opt_params["cnvg_test_mode"] == "P_update":
            converged, cnvg_test_vals = cnvg_check(iter, abs_P_diff_sum, cnvg_test_vals, opt_params)
        elif opt_params["cnvg_test_mode"] == "MCP_diff":
            num_LCPs = 1
            MCP =  strat_comp.compute_LCPs(P, F0, tau, num_LCPs)
            MCP_diff = MCP - old_MCP
            converged, cnvg_test_vals = cnvg_check(iter, MCP_diff, cnvg_test_vals, opt_params)
            old_MCP = MCP
        # terminate optimization if maximum iteration count reached:
        if iter == opt_params["max_iters"]:
            converged = True
        iter = iter + 1

    # convergence or max iteration count has been reached...
    F = strat_comp.compute_cap_probs(P, F0, tau)
    final_MCP = jnp.min(F)
    tracked_vals["final_MCP"].append(final_MCP)
    print("Minimum Capture Probability at iteration " + str(iter) + ":")
    print(final_MCP)
    tracked_vals["final_iters"].append(iter)
    return P, F, tracked_vals

def setup_optimizer(opt_params):
    """
    Initialize Optax optimizer from given optimization parameters.

    Parameters
    ----------
    opt_params : dict
        Optimization parameters. 
    
    Returns
    ----------
    Optax GradientTransformation
        Optax optimizer with specified parameters. 
    """
    # create learning rate schedule
    if opt_params["lr_schedule_type"] == "constant":
        schedule = optax.constant_schedule(opt_params["scaled_learning_rate"])
    elif opt_params["lr_schedule_type"] == "piecewise":
        # convert keys of map from string to int
        boundaries_and_scales = {int(key): value for key, value in opt_params["lr_schedule_args"].items()}
        schedule = optax.piecewise_constant_schedule(opt_params["scaled_learning_rate"], boundaries_and_scales)
    elif opt_params["lr_schedule_type"] == "exponential":
        schedule = optax.exponential_decay(opt_params["scaled_learning_rate"], *opt_params["lr_schedule_args"])
    else:
        raise ValueError("Invalid value. Acceptable values include: `constant`, `piecewise`, `exponential`.")
    
    # instantiate optimizer
    if opt_params["optimizer_name"] == "sgd":
        if opt_params["use_momentum"]:
            optimizer = optax.sgd(schedule, momentum=opt_params["mom_decay_rate"], nesterov=opt_params["use_nesterov"])
        else:
            optimizer = optax.sgd(schedule)
    elif opt_params["optimizer_name"] == "adagrad":
        optimizer = optax.adagrad(schedule)
    elif opt_params["optimizer_name"] == "adam":
        optimizer = optax.adam(schedule)
    elif opt_params["optimizer_name"] == "rmsprop":
        if opt_params["use_momentum"]:
            optimizer = optax.rmsprop(schedule, momentum=opt_params["mom_decay_rate"], nesterov=opt_params["use_nesterov"])
        else:
            optimizer = optax.rmsprop(schedule)
    else:
        raise ValueError("Invalid value. Acceptable values include: `sgd`, `adagrad`, `adam`, `rmsprop`.")
    return optimizer

def cnvg_check(iter, new_val, cnvg_test_vals, opt_params):
    """
    Check for convergence based on trailing average of chosen metric.

    Parameters
    ----------
    iter : int
        Current iteration number of optimzation process. 
    new_val
        Value of convergence metric for most recent iteration. 
    cnvg_test_vals : deque
        Set of convergence metric values from recent iterations. 
    opt_params : dict
        Optimization parameters.
    
    Returns
    ----------
    bool
        Flag indicating whether convergence has been reached.
    deque
        Updated set of convergence metric values.
    """
    cnvg_test_vals.append(new_val)
    if iter > opt_params["cnvg_window_size"]:
        cnvg_test_vals.popleft()
    MA_val = np.mean(cnvg_test_vals)
    if iter > opt_params["cnvg_window_size"]:
        converged = MA_val < opt_params["cnvg_radius"]
    else:
        converged = False
    if iter == opt_params["max_iters"]:
        converged = True
    return converged, cnvg_test_vals

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    # test_set_name = "Quick_Setup_Test"
    # test_spec = TestSpec(test_spec_filepath=os.getcwd() + "/robosurvstratopt/test_specs/quick_test_spec.json")

    test_set_name = "Default_Setup_Test"
    # test_spec = TestSpec(test_spec_filepath=os.getcwd() + "/robosurvstratopt/test_specs/default_test_spec.json")
    # test_spec = TestSpec(test_spec_filepath=os.getcwd() + "/robosurvstratopt/test_specs/default_test_spec_2.json")
    test_spec = TestSpec(test_spec_filepath=os.getcwd() + "/robosurvstratopt/test_specs/default_test_spec_3.json")

    test_set_start_time = time.time()
    run_test_set(test_set_name, test_spec)
    print("Running test_set_" + test_set_name + " took " + str(time.time() - test_set_start_time) + " seconds to complete.")

