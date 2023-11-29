# Optimization of the performance of stochastic surveillance strategies
from collections import deque
import json
import os
import shutil
import time

import functools
import jax
import jax.numpy as jnp
import numpy as np
import optax
# import csv

import graph_comp
import strat_comp
import strat_viz
from test_spec import TestSpec

def run_test_set(problem_set):
    # run all tests defined in test spec:
    run_times = []
    for problem in problem_set:
        times = run_test(problem)
        run_times.append(times)

def run_test(problem):    
    cnvg_times = []
    for k in range(problem.problem_params["num_init_Ps"]):
        print("Optimizing with initial P matrix number " + str(k + 1) + "...")
        print("Using optimizer: " + problem.opt_params["optimizer_name"])
        # setup optimizer (set learning rate)
        problem.set_learning_rate()

        start_time = time.time()
        run_optimizer(problem)
        cnvg_time = time.time() - start_time
        cnvg_times.append(cnvg_time)
        print("--- Optimization took: %s seconds ---" % (cnvg_time))
    return cnvg_times

def run_optimizer(problem):
    check_time = time.time()
    P0 = problem.init_rand_P()
    optimizer = setup_optimizer(problem.opt_params)
    opt_state = optimizer.init(P0)
    cnvg_test_vals = deque()  # queue storing recent values of desired metric, for checking convergence
    # convert keys of map from string to int
    num_LCPs_schedule = {int(key): value for key, value in problem.opt_params["num_LCPs_schedule"].items()}
    num_LCPs_schedule = optax.piecewise_constant_schedule(problem.opt_params["num_init_LCPs"], num_LCPs_schedule)

    @functools.partial(jax.jit, static_argnames=['num_LCPs'])
    def step(Q, P, MCP, loss, opt_state, num_LCPs):
        P_old = P
        loss_old = loss
        # gradient computation (should we be using value_and_grad here?)
        grad = problem.compute_gradient()
        loss = problem.compute_loss()
        updates, opt_state = optimizer.update(grad, opt_state)
        Q = optax.apply_updates(Q, updates)
        P = problem.apply_parametrization(Q)
        P_diff = P - P_old
        abs_P_diff_sum = jnp.sum(jnp.abs(P_diff))
        loss_diff = jnp.abs((loss - loss_old)/loss_old)

        if opt_params["cnvg_test_mode"] == "MCP_diff":
            MCP_old = MCP
            MCP = strat_comp.compute_LCPs(P, F0, tau)
            MCP_diff = MCP - MCP_old
        else:
            MCP = MCP_diff = jnp.nan

        return Q, P, P_diff, abs_P_diff_sum, MCP, MCP_diff, loss, loss_diff, opt_state
    
    # Run gradient-based optimization process:
    iter = 0 
    converged = False
    while not converged:
        num_LCPs = int(num_LCPs_schedule(iter))
        # apply update to P matrix, and parametrization Q:
        Q, P, P_diff, abs_P_diff_sum, MCP, MCP_diff, loss, loss_diff, opt_state = step(Q, P, MCP, loss, opt_state, num_LCPs)
        # check for convergence:
        converged = problem.cnvg_check(iter, abs_P_diff_sum, MCP_diff, loss_diff)
        iter = iter + 1

    print("*************************")
    print("FINAL ITER = " + str(iter))
    print("FINAL LOSS = " + str(loss))
    return P

def setup_optimizer(opt_params):
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

if __name__ == '__main__':
