# Optimization of the performance of stochastic surveillance strategies
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from patrol_problem import PatrolProblem

def run_test_set(problem_set):
    # run all tests defined in test spec:
    run_times = []
    for problem in problem_set:
        times = run_test(problem)
        run_times.append(times)

def run_test(problem: PatrolProblem):
    problem.initialize()
    cnvg_times = []
    Qs = problem.init_rand_Ps()
    for k in range(problem.opt_params["num_init_Ps"]):
        print("Optimizing with initial P matrix number " + str(k + 1) + "...")
        print("Using optimizer: " + problem.opt_params["optimizer_name"])
        Q = Qs[k]
        problem.set_learning_rate(Q)
        start_time = time.time()
        run_optimizer(problem, Q)
        cnvg_time = time.time() - start_time
        cnvg_times.append(cnvg_time)
        print("--- Optimization took: %s seconds ---" % (cnvg_time))
    return cnvg_times

def run_optimizer(problem : PatrolProblem, Q):
    optimizer = setup_optimizer(problem.opt_params)
    opt_state = optimizer.init(Q)

    @jax.jit
    def step(Q, P, loss, opt_state):
        Q_old = Q
        P_old = P
        loss_old = loss
        (loss, grad) = problem.compute_loss_and_gradient(Q)
        updates, opt_state = optimizer.update(grad, opt_state)
        Q = optax.apply_updates(Q, updates)
        P = problem.apply_parametrization(Q)
        abs_P_diff_sum = jnp.sum(jnp.abs(P - P_old))
        loss_diff = jnp.abs((loss - loss_old)/loss_old)
        return Q, P, abs_P_diff_sum, loss, loss_diff, opt_state
    
    # Run gradient-based optimization process:
    iter = 0 
    P = Q
    loss = 1000
    converged = False
    # for i in range(10):
    while not converged:
        # apply update to P matrix, and parametrization Q:
        Q, P, abs_P_diff_sum, loss, loss_diff, opt_state = step(Q, P, loss, opt_state)
        # track metrics:
        problem.update_metrics_tracking(P)
        # check for convergence:
        converged = problem.cnvg_check(iter, abs_P_diff_sum, loss_diff)
        iter = iter + 1
    problem.save_tracked_metrics()

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
    # fn = test_spec_filepath= os.getcwd() + "/robosurvstratopt/problem_specs/demo_problem_spec_1.json"
    # fn = test_spec_filepath= os.getcwd() + "/robosurvstratopt/problem_specs/demo_problem_spec_2.json"
    fn = test_spec_filepath= os.getcwd() + "/robosurvstratopt/problem_specs/demo_problem_spec_3.json"
    results_directory = os.getcwd() + "/results/local/demo_1"
    with open(fn, "r") as problem_spec_file:
        json_string = problem_spec_file.read()
        problem_spec_dict = json.loads(json_string)
        name = problem_spec_dict["problem_spec_name"]
        problem_params = problem_spec_dict["problem_params"]
        opt_params = problem_spec_dict["optimizer_params"]
        problem_spec = PatrolProblem(name, problem_params, opt_params, results_directory)
        run_test(problem_spec)
