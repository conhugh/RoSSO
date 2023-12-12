# Computation of quantities relevant to optimization of stochastic surveillance strategies
import functools
import itertools
import jax
import jax.numpy as jnp
import numpy as np
from jax import jacrev, jit
import graph_comp

def init_rand_Ps(A, num, seed=0):
    """
    Generate a set of `num` random initial transition probability matrices using PRNG key `seed`.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph.
    num : int 
        Number of initial transition probability matrices to generate.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Set of `num` unique, valid, random initial transition probability matrices. 
    """
    A_shape = jnp.shape(A)
    key = jax.random.PRNGKey(seed)
    initPs = jnp.zeros((A_shape[0], A_shape[1], num),  dtype='float32')
    for k in range(num):
        key, subkey = jax.random.split(key)
        P0 = A*jax.random.uniform(subkey, A_shape)
        P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0) 
        initPs = initPs.at[:, : , k].set(P0)
    return initPs

def multi_init_rand_Ps(As, N, num, seed=0):
    A_shape = jnp.shape(As[0, :, :])
    key = jax.random.PRNGKey(seed)
    # initPs = jnp.zeros((A_shape[0], A_shape[1], N, num),  dtype='float32')
    initPs = jnp.zeros((num, N, A_shape[0], A_shape[1]),  dtype='float32')
    for k in range(num):
        for i in range(N):
            key, subkey = jax.random.split(key)
            P0 = As[i, :, :]*jax.random.uniform(subkey, A_shape)
            P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0) 
            # initPs = initPs.at[:, :, i, k].set(P0)
            initPs = initPs.at[k, i, :, :].set(P0)
    return initPs

def oop_init_rand_Ps(A, N, num, key):
    # current implementation assumes same adjacency matrix for each robot
    if N > 1:
        A = A[0, :, :]
    A_shape = jnp.shape(A)
    initPs = []
    for _ in range(num):
        initP = jnp.zeros((N, A_shape[0], A_shape[1]),  dtype='float32')
        for i in range(N):
            key, subkey = jax.random.split(key)
            P0 = A*jax.random.uniform(subkey, A_shape)
            P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)
            initP = initP.at[i, :, :].set(P0)
        initPs.append(initP)
    return initPs

@functools.partial(jit, static_argnames=['use_abs_param'])
def comp_P_param(Q, A, use_abs_param=True):
    P = Q*A
    if use_abs_param:
        P = jnp.abs(P) # apply component-wise absolute-value
    else:
        P = jnp.maximum(jnp.zeros_like(P), P) # apply component-wise ReLU   
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize rows to generate valid prob dist 
    return P

def comp_pi_penalty(P, pi, alpha):
    n = len(pi)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi)))
    return alpha*penalty

def comp_multi_pi_penalty(Ps, pi, alpha):
    N = Ps.shape[0]
    n = Ps.shape[-1]
    pi_Ps = (1/n)*jnp.ones((N, n))
    init_vals = (Ps, pi_Ps)
    (_, pi_Ps) = jax.lax.fori_loop(0, 10, power_iteration, init_vals)
    pi_Ps_avg = jnp.mean(pi_Ps, axis=0)
    return alpha*jnp.linalg.norm(jnp.array(pi) - pi_Ps_avg)

def weighted_FHTs_loop_body(k, loop_vals):
    F_mats, P, D_idx, W, w_max = loop_vals
    n = jnp.shape(P)[0]
    P_direct = P*(W == k+1)
    idx = (D_idx[k] + w_max).astype(int)
    D_k = F_mats[jnp.ravel(idx), jnp.tile(jnp.arange(n), n), :]
    D_k = jnp.reshape(D_k, (n, n, n))
    D_k = D_k.at[:, jnp.arange(n), jnp.arange(n)].set(0)
    multi_step_probs = jnp.matmul(P, D_k)
    multi_step_probs = multi_step_probs[jnp.arange(n), jnp.arange(n), :]
    F_mats = F_mats.at[k + w_max, :, :].set(P_direct + multi_step_probs)
    return (F_mats, P, D_idx, W, w_max)

def precompute_multi(n, N):
    combs = jnp.array(list(itertools.product(range(n), repeat=N+1)))
    combs_len = len(combs)
    return combs, combs_len

############################################################
# Stackelberg formulation
############################################################
@functools.partial(jit, static_argnames=['tau'])
def compute_cap_probs(P, F0, tau):
    """
    Compute Capture Probability Matrix.

    Parameters
    ----------
    P : jaxlib.xla_extension.DeviceArray 
        Transition probability matrix.
    F0 : jaxlib.xla_extension.DeviceArray 
        Placeholder to be populated with FHT Probability matrices. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Capture Probability matrix. 
    
    See Also
    --------
    compute_FHT_probs
    """
    F0 = F0.at[:, :, 0].set(P)
    for i in range(1, tau):
        F0 = F0.at[:, :, i].set(jnp.matmul(P, (F0[:, :, i - 1] - jnp.diag(jnp.diag(F0[:, :, i - 1])))))
    cap_probs = jnp.sum(F0, axis=2)
    return cap_probs

@functools.partial(jit, static_argnames=['tau', 'num_LCPs'])
def compute_LCPs(P, F0, tau, num_LCPs=1):
    """
    Compute Lowest `num_LCPs` Capture Probabilities.

    Parameters
    ----------
    P : jaxlib.xla_extension.DeviceArray 
        Transition probability matrix. 
    F0 : jaxlib.xla_extension.DeviceArray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    num_LCPs : int
        Number of the lowest capture probabilities to compute. 
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Set of `num_LCPs` lowest capture probabilities. 
    
    See Also
    --------
    compute_cap_probs
    """
    cap_probs = compute_cap_probs(P, F0, tau)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        F_vec = cap_probs.flatten('F')
        lcps = jnp.sort(F_vec)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau', 'num_LCPs', 'use_abs_param'])
def loss_LCP(Q, A, F0, tau, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_LCPs(P, F0, tau, num_LCPs)
    return jnp.mean(lcps)

# Autodiff parametrized loss function
_comp_LCP_grad = jacrev(loss_LCP)
@functools.partial(jit, static_argnames=['tau', 'num_LCPs', 'use_abs_param'])
def comp_avg_LCP_grad(Q, A, F0, tau, num_LCPs=1, use_abs_param=True):
    grad = _comp_LCP_grad(Q, A, F0, tau, num_LCPs, use_abs_param) 
    return grad

# pi must be a tuple
@functools.partial(jit, static_argnames=['tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def loss_LCP_pi(Q, A, F0, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_LCPs(P, F0, tau, num_LCPs)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return jnp.mean(lcps) - alpha*penalty

_comp_avg_LCP_pi_grad = jacrev(loss_LCP_pi)
@functools.partial(jit, static_argnames=['tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def comp_avg_LCP_pi_grad(Q, A, F0, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    grad = _comp_avg_LCP_pi_grad(Q, A, F0, tau, pi, alpha, num_LCPs, use_abs_param)
    return grad

############################################################
# Heterogeneous Attack Duration Stackelberg formulation
############################################################
# input tau_vec must be a tuple
@functools.partial(jit, static_argnames=['tau_vec'])
def compute_hetero_tau_cap_probs(P, F0, tau_vec):
    n = jnp.shape(P)[0]
    tau_max = max(tau_vec)
    F0 = F0.at[:, :, 0].set(P)
    for i in range(1, tau_max):
        F0 = F0.at[:, :, i].set(jnp.matmul(P, (F0[:, :, i - 1] - jnp.diag(jnp.diag(F0[:, :, i - 1])))))
    cap_probs = jnp.zeros((n, n))
    for i in range(n):
        cap_probs = cap_probs.at[:, i].set(jnp.sum(F0[:, i, :tau_vec[i]], axis=1))
    return cap_probs

@functools.partial(jit, static_argnames=['tau_vec', 'num_LCPs'])
def compute_hetero_tau_LCPs(P, F0, tau_vec, num_LCPs=1):
    cap_probs = compute_hetero_tau_cap_probs(P, F0, tau_vec)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        F_vec = cap_probs.flatten('F')
        lcps = jnp.sort(F_vec)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau_vec', 'num_LCPs', 'use_abs_param'])
def loss_hetero_tau_LCP(Q, A, F0, tau_vec, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_hetero_tau_LCPs(P, F0, tau_vec, num_LCPs)
    return jnp.mean(lcps)

# Autodiff parametrized loss function
_comp_hetero_tau_LCP_grad = jacrev(loss_hetero_tau_LCP)
@functools.partial(jit, static_argnames=['tau_vec', 'num_LCPs', 'use_abs_param'])
def comp_avg_hetero_tau_LCP_grad(Q, A, F0, tau_vec, num_LCPs=1, use_abs_param=True):
    grad = _comp_hetero_tau_LCP_grad(Q, A, F0, tau_vec, num_LCPs, use_abs_param) 
    return grad

# pi must be a tuple
@functools.partial(jit, static_argnames=['tau_vec', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def loss_hetero_tau_LCP_pi(Q, A, F0, tau_vec, pi, alpha, num_LCPs=1, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_hetero_tau_LCPs(P, F0, tau_vec, num_LCPs)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return jnp.mean(lcps) - alpha*penalty

_comp_avg_hetero_tau_LCP_pi_grad = jacrev(loss_hetero_tau_LCP_pi)
@functools.partial(jit, static_argnames=['tau_vec', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def comp_avg_hetero_tau_LCP_pi_grad(Q, A, F0, tau_vec, pi, alpha, num_LCPs=1, use_abs_param=True):
    grad = _comp_avg_hetero_tau_LCP_pi_grad(Q, A, F0, tau_vec, pi, alpha, num_LCPs, use_abs_param)
    return grad

############################################################
# Stackelberg Co-Optimization formulation
############################################################
# B: defense budget
@functools.partial(jit, static_argnames=['B'])
def greedy_co_opt_cap_probs(P, F0, B):
    n = jnp.shape(P)[0]
    tau_max = B - n + 1
    F0 = F0.at[:, :, 0].set(P)
    for i in range(1, tau_max):
        F0 = F0.at[:, :, i].set(jnp.matmul(P, (F0[:, :, i - 1] - jnp.diag(jnp.diag(F0[:, :, i - 1])))))

    cap_probs = P
    tau_vec = jnp.ones(n)
    B -= n
    while B > 0:
        min_idx = jnp.argmin(cap_probs)
        _, col_idx = divmod(min_idx, n)
        tau_vec = tau_vec.at[col_idx].set(tau_vec[col_idx] + 1)
        cap_probs = cap_probs.at[:, col_idx].set(cap_probs[:, col_idx] + F0[:, col_idx, (tau_vec[col_idx]-1).astype(int)])
        B -= 1
    
    return tau_vec, cap_probs

@functools.partial(jit, static_argnames=['B', 'num_LCPs'])
def compute_greedy_co_opt_LCPs(P, F0, B, num_LCPs=1):
    _, cap_probs = greedy_co_opt_cap_probs(P, F0, B)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        F_vec = cap_probs.flatten('F')
        lcps = jnp.sort(F_vec)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['B', 'num_LCPs', 'use_abs_param'])
def loss_greedy_co_opt_LCP(Q, A, F0, B, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_greedy_co_opt_LCPs(P, F0, B, num_LCPs)
    return jnp.mean(lcps)

# Autodiff parametrized loss function
_comp_greedy_co_opt_LCP_grad = jacrev(loss_greedy_co_opt_LCP)
@functools.partial(jit, static_argnames=['B', 'num_LCPs', 'use_abs_param'])
def comp_avg_greedy_co_opt_LCP_grad(Q, A, F0, B, num_LCPs=1, use_abs_param=True):
    grad = _comp_greedy_co_opt_LCP_grad(Q, A, F0, B, num_LCPs, use_abs_param) 
    return grad

# pi must be a tuple
@functools.partial(jit, static_argnames=['B', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def loss_greedy_co_opt_LCP_pi(Q, A, F0, B, pi, alpha, num_LCPs=1, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_greedy_co_opt_LCPs(P, F0, B, num_LCPs)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return jnp.mean(lcps) - alpha*penalty

_comp_avg_greedy_co_opt_LCP_pi_grad = jacrev(loss_greedy_co_opt_LCP_pi)
@functools.partial(jit, static_argnames=['B', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def comp_avg_greedy_co_opt_LCP_pi_grad(Q, A, F0, B, pi, alpha, num_LCPs=1, use_abs_param=True):
    grad = _comp_avg_greedy_co_opt_LCP_pi_grad(Q, A, F0, B, pi, alpha, num_LCPs, use_abs_param)
    return grad

############################################################
# Weighted Stackelberg formulation
############################################################
def precompute_weighted_Stackelberg(W, w_max, tau):
    n = jnp.shape(W)[0]
    D_idx = jnp.zeros((tau, n, n))
    for k in range(tau):
        for i in range(n):
            vec = jnp.where(W[i, :] > 0, k - W[i, :], -w_max)
            D_idx = D_idx.at[k, i].set(vec)
    return D_idx

@functools.partial(jit, static_argnames=['w_max', 'tau'])
def compute_weighted_cap_probs(P, D_idx, W, w_max, tau):
    n = jnp.shape(P)[0]
    F_mats = jnp.zeros((tau + w_max, n, n))
    init_vals = (F_mats, P, D_idx, W, w_max)
    F_mats, _, _, _, _ = jax.lax.fori_loop(0, tau, weighted_FHTs_loop_body, init_vals)
    F_mats = F_mats[w_max:, :, :]
    cap_probs = jnp.sum(F_mats, axis=0)
    return jnp.ravel(cap_probs, order='F')

@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs'])
def compute_weighted_LCPs(P, D_idx, W, w_max, tau, num_LCPs=1):
    cap_probs = compute_weighted_cap_probs(P, D_idx, W, w_max, tau)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        lcps = jnp.sort(cap_probs)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def loss_weighted_LCP(Q, A, D_idx, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_weighted_LCPs(P, D_idx, W, w_max, tau, num_LCPs)
    return jnp.mean(lcps)

# Autodiff parametrized loss function
_comp_weighted_LCP_grad = jacrev(loss_weighted_LCP)
@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def comp_avg_weighted_LCP_grad(Q, A, D_idx, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    grad = _comp_weighted_LCP_grad(Q, A, D_idx, W, w_max, tau, num_LCPs, use_abs_param)
    return grad

# pi must be a tuple
@functools.partial(jit, static_argnames=['w_max', 'tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def loss_weighted_LCP_pi(Q, A, D_idx, W, w_max, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_weighted_LCPs(P, D_idx, W, w_max, tau, num_LCPs)
    n = len(pi)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return jnp.mean(lcps) - alpha*penalty

_comp_avg_weighted_LCP_pi_grad = jacrev(loss_weighted_LCP_pi)
@functools.partial(jit, static_argnames=['w_max', 'tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def comp_avg_weighted_LCP_pi_grad(Q, A, D_idx, W, w_max, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    grad = _comp_avg_weighted_LCP_pi_grad(Q, A, D_idx, W, w_max, tau, pi, alpha, num_LCPs, use_abs_param)
    return grad

############################################################
# Weighted Stackelberg Co-Optimization formulation
############################################################
def precompute_weighted_Stackelberg_co_opt(W, w_max, B):
    n = jnp.shape(W)[0]
    tau_max = B - n + 1
    D_idx = jnp.zeros((tau_max, n, n))
    for k in range(tau_max):
        for i in range(n):
            vec = jnp.where(W[i, :] > 0, k - W[i, :], -w_max)
            D_idx = D_idx.at[k, i].set(vec)
    return D_idx

def greedy_co_opt_loop_body(_, loop_vals):
    tau_vec, cap_probs, F0 = loop_vals
    n = len(tau_vec)
    min_idx = jnp.argmin(cap_probs)
    _, col_idx = divmod(min_idx, n)
    tau_vec = tau_vec.at[col_idx].set(tau_vec[col_idx] + 1)
    cap_probs = cap_probs.at[:, col_idx].set(cap_probs[:, col_idx] + F0[(tau_vec[col_idx]-1).astype(int), :, col_idx])
    return (tau_vec, cap_probs, F0)

@functools.partial(jit, static_argnames=['w_max', 'B'])
def greedy_co_opt_weighted_cap_probs(P, D_idx, W, w_max, B):
    n = jnp.shape(P)[0]
    tau_max = B - n + 1
    F_mats = jnp.zeros((tau_max + w_max, n, n))
    init_vals = (F_mats, P, D_idx, W, w_max)
    F_mats, _, _, _, _ = jax.lax.fori_loop(0, tau_max, weighted_FHTs_loop_body, init_vals)
    F0 = F_mats[w_max:, :, :]

    tau_vec = jnp.zeros(n)
    cap_probs = jnp.zeros((n, n))
    init_vals = (tau_vec, cap_probs, F0)
    tau_vec, cap_probs, _ = jax.lax.fori_loop(0, B, greedy_co_opt_loop_body, init_vals)
    
    return tau_vec, cap_probs

@functools.partial(jit, static_argnames=['w_max', 'B', 'num_LCPs'])
def compute_greedy_co_opt_weighted_LCPs(P, D_idx, W, w_max, B, num_LCPs=1):
    _, cap_probs = greedy_co_opt_weighted_cap_probs(P, D_idx, W, w_max, B)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        F_vec = cap_probs.flatten('F')
        lcps = jnp.sort(F_vec)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['w_max', 'B', 'num_LCPs', 'use_abs_param'])
def loss_greedy_co_opt_weighted_LCP(Q, A, D_idx, W, w_max, B, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_greedy_co_opt_weighted_LCPs(P, D_idx, W, w_max, B, num_LCPs)
    return jnp.mean(lcps)

# Autodiff parametrized loss function
_comp_greedy_co_opt_weighted_LCP_grad = jacrev(loss_greedy_co_opt_weighted_LCP)
@functools.partial(jit, static_argnames=['w_max', 'B', 'num_LCPs', 'use_abs_param'])
def comp_avg_greedy_co_opt_weighted_LCP_grad(Q, A, D_idx, W, w_max, B, num_LCPs=1, use_abs_param=True):
    grad = _comp_greedy_co_opt_weighted_LCP_grad(Q, A, D_idx, W, w_max, B, num_LCPs, use_abs_param) 
    return grad

# pi must be a tuple
@functools.partial(jit, static_argnames=['w_max', 'B', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def loss_greedy_co_opt_weighted_LCP_pi(Q, A, D_idx, W, w_max, B, pi, alpha, num_LCPs=1, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_greedy_co_opt_weighted_LCPs(P, D_idx, W, w_max, B, num_LCPs)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return jnp.mean(lcps) - alpha*penalty

_comp_avg_greedy_co_opt_weighted_LCP_pi_grad = jacrev(loss_greedy_co_opt_weighted_LCP_pi)
@functools.partial(jit, static_argnames=['w_max', 'B', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def comp_avg_greedy_co_opt_weighted_LCP_pi_grad(Q, A, D_idx, W, w_max, B, pi, alpha, num_LCPs=1, use_abs_param=True):
    grad = _comp_avg_greedy_co_opt_weighted_LCP_pi_grad(Q, A, D_idx, W, w_max, B, pi, alpha, num_LCPs, use_abs_param)
    return grad

############################################################
# Multi-Agent Stackelberg formulation
############################################################
@functools.partial(jit, static_argnames=['tau'])
def compute_multi_cap_probs(Ps, F0s, combs, tau):
    n = jnp.shape(Ps)[0]
    N = Ps.shape[2]
    for r in range(N):
        F0s = F0s.at[:, :, 0, r].set(Ps[:, :, r])
        for i in range(1, tau):
            F0s = F0s.at[:, :, i, r].set(jnp.matmul(Ps, (F0s[:, :, i - 1, r] - jnp.diag(jnp.diag(F0s[:, :, i - 1, r])))))
        indiv_cap_probs = jnp.sum(F0s, axis=2)

    cap_probs = jnp.zeros(len(combs))
    for i in range(len(combs)):
        idx_vec = combs[i]
        not_cap_prob = jnp.prod(1 - indiv_cap_probs[idx_vec[:-1], idx_vec[-1], jnp.arange(N)])
        cap_probs = cap_probs.at[i].set(1 - not_cap_prob) 

    return jnp.reshape(cap_probs, (n**N, n), order='F')

@functools.partial(jit, static_argnames=['tau', 'num_LCPs'])
def compute_multi_LCPs(Ps, F0s, tau, num_LCPs=1):
    cap_probs = compute_multi_cap_probs(Ps, F0s, tau)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        lcps = jnp.sort(cap_probs)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau', 'num_LCPs', 'use_abs_param'])
def loss_multi_LCP(Qs, As, F0s, tau, num_LCPs=1, use_abs_param=True):
    N = Qs.shape[2]
    Ps = jnp.zeros_like(Qs)
    for i in range(N):
        P = comp_P_param(Qs[:, :, i], As[:, :, i], use_abs_param)
        Ps = Ps.at[:, :, i].set(P)
    return jnp.mean(compute_multi_LCPs(Ps, F0s, tau, num_LCPs))

# Autodiff parametrized loss function
_comp_multi_LCP_grad = jacrev(loss_multi_LCP)
@functools.partial(jit, static_argnames=['tau', 'num_LCPs', 'use_abs_param'])
def comp_avg_multi_LCP_grad(Qs, As, F0s, tau, num_LCPs=1, use_abs_param=True):
    grad = _comp_multi_LCP_grad(Qs, As, F0s, tau, num_LCPs, use_abs_param)
    return grad

############################################################
# Weighted Multi-Agent Stackelberg formulation
############################################################
def weighted_multi_comb_loop_body(i, loop_vals):
    cap_probs, combs, indiv_cap_probs = loop_vals
    N = jnp.shape(indiv_cap_probs)[0]
    idx_vec = combs[i]
    not_cap_prob = jnp.prod(1 - indiv_cap_probs[jnp.arange(N), idx_vec[:-1], idx_vec[-1]])
    cap_probs = cap_probs.at[i].set(1 - not_cap_prob) 
    return (cap_probs, combs, indiv_cap_probs)

@functools.partial(jit, static_argnames=['N', 'combs_len', 'w_max', 'tau'])
def compute_weighted_multi_cap_probs(Ps, D_idx, combs, N, combs_len, W, w_max, tau):
    n = jnp.shape(Ps)[-1]
    def loop_body(r, F_mats_multi):
        F_mats = F_mats_multi[r, :, :, :]
        init_vals = (F_mats, Ps[r, :, :], D_idx, W, w_max)
        F_mats, _, _, _, _ = jax.lax.fori_loop(0, tau, weighted_FHTs_loop_body, init_vals)
        F_mats_multi = F_mats_multi.at[r, :, :, :].set(F_mats)
        return F_mats_multi

    F_mats_multi = jnp.zeros((N, tau + w_max, n, n))
    F_mats_multi = jax.lax.fori_loop(0, N, loop_body, F_mats_multi)
    F_mats_multi = F_mats_multi[:, w_max:, :, :]
    indiv_cap_probs = jnp.sum(F_mats_multi, axis=1)

    cap_probs = jnp.zeros(combs_len)
    init_vals = (cap_probs, combs, indiv_cap_probs)
    cap_probs, _, _ = jax.lax.fori_loop(0, combs_len, weighted_multi_comb_loop_body, init_vals)
    return jnp.reshape(cap_probs, (n**N, n))

@functools.partial(jit, static_argnames=['N', 'combs_len', 'w_max', 'tau', 'num_LCPs'])
def compute_weighted_multi_LCPs(Ps, D_idx, combs, N, combs_len, W, w_max, tau, num_LCPs=1):
    cap_probs = compute_weighted_multi_cap_probs(Ps, D_idx, combs, N, combs_len, W, w_max, tau)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        lcps = jnp.sort(cap_probs)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['N', 'combs_len', 'w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def loss_weighted_multi_LCP(Qs, As, D_idx, combs, N, combs_len, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    N = Qs.shape[0]
    Ps = jnp.zeros_like(Qs)
    for i in range(N):
        P = comp_P_param(Qs[i, :, :], As[i, :, :], use_abs_param)
        Ps = Ps.at[i, :, :].set(P)
    return jnp.mean(compute_weighted_multi_LCPs(Ps, D_idx, combs, N, combs_len, W, w_max, tau, num_LCPs))

# Autodiff parametrized loss function
_comp_weighted_multi_LCP_grad = jacrev(loss_weighted_multi_LCP)
@functools.partial(jit, static_argnames=['N', 'combs_len', 'w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def comp_avg_weighted_multi_LCP_grad(Qs, As, D_idx, combs, N, combs_len, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    grad = _comp_weighted_multi_LCP_grad(Qs, As, D_idx, combs, N, combs_len, W, w_max, tau, num_LCPs, use_abs_param)
    return grad

def power_iteration(_, loop_vals):
    (Ps, pi_Ps) = loop_vals
    pi_Ps_new = jnp.dot(pi_Ps, Ps)[0]
    row_sums = jnp.sum(pi_Ps_new, axis=1, keepdims=True)
    pi_Ps = pi_Ps_new / row_sums
    return (Ps, pi_Ps)

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['N', 'combs_len', 'w_max', 'tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def loss_weighted_multi_LCP_pi(Qs, As, D_idx, combs, N, combs_len, W, w_max, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    N = Qs.shape[0]
    n = Qs.shape[-1]
    Ps = jnp.zeros_like(Qs)
    for i in range(N):
        P = comp_P_param(Qs[i, :, :], As[i, :, :], use_abs_param)
        Ps = Ps.at[i, :, :].set(P)
    mu = jnp.mean(compute_weighted_multi_LCPs(Ps, D_idx, combs, N, combs_len, W, w_max, tau, num_LCPs))

    pi_Ps = (1/n)*jnp.ones((N, n))
    init_vals = (Ps, pi_Ps)
    (_, pi_Ps) = jax.lax.fori_loop(0, 10, power_iteration, init_vals)
    pi_Ps_avg = jnp.mean(pi_Ps, axis=0)

    return mu - alpha*jnp.sum((jnp.array(pi) - pi_Ps_avg)**2)

# Autodiff parametrized loss function
_comp_weighted_multi_LCP_pi_grad = jacrev(loss_weighted_multi_LCP_pi)
@functools.partial(jit, static_argnames=['N', 'combs_len', 'w_max', 'tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def comp_avg_weighted_multi_LCP_pi_grad(Qs, As, D_idx, combs, N, combs_len, W, w_max, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    grad = _comp_weighted_multi_LCP_pi_grad(Qs, As, D_idx, combs, N, combs_len, W, w_max, tau, pi, alpha, num_LCPs, use_abs_param)
    return grad

############################################################
# Mean Hitting Time formulation
############################################################
@jit
def compute_MHT(P):
    eigs = jnp.linalg.eigvals(P)
    sorted_eigs = eigs[jnp.argsort(jnp.abs(eigs))]
    m = 1 + jnp.sum(1 / (1 - sorted_eigs[:-1]))
    return jnp.real(m)

@functools.partial(jit, static_argnames=['pi'])
def compute_MHT_GPU(P, pi):
    n = len(pi)
    pi = jnp.array(pi)
    M_d = jnp.diag(1 / pi)
    lhs = jnp.identity(n) - P
    lhs = jnp.hstack((lhs, jnp.zeros((n, n**2 - n))))
    LHS = lhs[1:, 1:n**2 - n + 1]
    for i in range(1, n):
        rows = jnp.vstack((lhs[:i, 0:n**2 - n + 1], lhs[i+1:, 0:n**2 - n + 1]))
        rows = jnp.hstack((rows[:, :i], rows[:, i+1:]))
        rows = jnp.hstack((rows[:, -i*(n-1):], rows[:, :-i*(n-1)]))
        LHS = jnp.vstack((LHS, rows))
    RHS = jnp.ones((n**2 - n, ))
    M_offdiag = jnp.linalg.solve(LHS, RHS)
    M_row = jnp.hstack((M_d[0, 0], M_offdiag[n-1::n-1]))
    return jnp.dot(M_row, pi)

@functools.partial(jit, static_argnames=['use_abs_param'])
def loss_MHT(Q, A, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    m = compute_MHT(P)
    return m

# Autodiff parametrized loss function
_comp_MHT_grad = jacrev(loss_MHT)
@functools.partial(jit, static_argnames=['use_abs_param'])
def comp_MHT_grad(Q, A, use_abs_param=True):
    grad = _comp_MHT_grad(Q, A, use_abs_param) 
    return grad

# pi must be a tuple
@functools.partial(jit, static_argnames=['pi', 'alpha', 'use_abs_param'])
def loss_MHT_pi(Q, A, pi, alpha, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    # m = compute_MHT(P)
    m = compute_MHT_GPU(P, pi)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return m + alpha*penalty

_comp_MHT_pi_grad = jacrev(loss_MHT_pi)
@functools.partial(jit, static_argnames=['pi', 'alpha', 'use_abs_param'])
def comp_MHT_pi_grad(Q, A, pi, alpha, use_abs_param=True):
    grad = _comp_MHT_pi_grad(Q, A, pi, alpha, use_abs_param) 
    return grad

############################################################
# Weighted Mean Hitting Time formulation
############################################################
# pi must be a tuple
@functools.partial(jit, static_argnames=['pi'])
def compute_weighted_MHT_pi(P, W, pi):
    n = P.shape[0]
    # m = compute_MHT_GPU(P, pi)
    m = compute_MHT(P)
    sclr = jnp.dot(jnp.array(pi),jnp.dot(P*jnp.array(W),jnp.ones((n,))))
    return jnp.squeeze(sclr*m)

# pi must be a tuple
@functools.partial(jit, static_argnames=['pi', 'alpha', 'use_abs_param'])
def loss_weighted_MHT_pi(Q, A, W, pi, alpha, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    m_w = compute_weighted_MHT_pi(P, W, pi)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return m_w + alpha*penalty

_comp_weighted_MHT_pi_grad = jacrev(loss_weighted_MHT_pi)
@functools.partial(jit, static_argnames=['pi', 'alpha', 'use_abs_param'])
def comp_weighted_MHT_pi_grad(Q, A, W, pi, alpha, use_abs_param=True):
    grad = _comp_weighted_MHT_pi_grad(Q, A, W, pi, alpha, use_abs_param) 
    return grad

############################################################
# Multi-Agent Mean Hitting Time formulation
############################################################
@jit
def compute_multi_MHT(Ps, combs):
    n = Ps.shape[0]
    N = Ps.shape[2]
    big_I = jnp.identity(n**(N+1))

    mat = jnp.identity(n)
    for i in range(N):
        mat = jnp.kron(mat, Ps[:, :, i])

    last_entry = combs[:, -1]
    other_entries = combs[:, :-1]
    e = jnp.any(last_entry[:, None] == other_entries, axis=1).astype(jnp.int32)

    big_mat = big_I - jnp.matmul(mat, big_I - jnp.diag(e))
    M_vec = jnp.linalg.solve(big_mat, jnp.ones(n**(N+1)))
    return jnp.reshape(M_vec, (n**N, n), order='F')

@functools.partial(jit, static_argnames=['use_abs_param'])
def loss_multi_MHT(Qs, As, use_abs_param=True):
    N = Qs.shape[2]
    Ps = jnp.zeros_like(Qs)
    for i in range(N):
        P = comp_P_param(Qs[:, :, i], As[:, :, i], use_abs_param)
        Ps = Ps.at[:, :, i].set(P)
    return jnp.mean(compute_multi_MHT(Ps))

# Autodiff parametrized loss function
_comp_multi_MHT_grad = jacrev(loss_multi_MHT)
@functools.partial(jit, static_argnames=['use_abs_param'])
def comp_multi_MHT_grad(Qs, As, use_abs_param=True):
    grad = _comp_multi_MHT_grad(Qs, As, use_abs_param) 
    return grad

############################################################
# Entropy Rate formulation
############################################################
@functools.partial(jit, static_argnames=['pi'])
def compute_ER_pi(P, pi):
    entropy_rate_matrix = P*jnp.log(jnp.where(P == 0, 1, P))
    entropy_rate = -jnp.dot(jnp.array(pi),jnp.sum(entropy_rate_matrix, axis=1))
    return jnp.squeeze(entropy_rate)

# pi must be a tuple
@functools.partial(jit, static_argnames=['pi', 'alpha', 'use_abs_param'])
def loss_ER_pi(Q, A, pi, alpha, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    h_r = compute_ER_pi(P, pi)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return h_r - alpha*penalty

_comp_ER_pi_grad = jacrev(loss_ER_pi)
@functools.partial(jit, static_argnames=['pi', 'alpha', 'use_abs_param'])
def comp_ER_pi_grad(Q, A, pi, alpha, use_abs_param=True):
    grad = _comp_ER_pi_grad(Q, A, pi, alpha, use_abs_param) 
    return grad

############################################################
# Return-Time Entropy formulation
############################################################
@functools.partial(jit, static_argnames=['pi', 'N_eta'])
def compute_RTE_pi(P, pi, N_eta):
    n = jnp.shape(P)[0]
    F_vecs = jnp.zeros((n**2, N_eta))
    F_vecs = F_vecs.at[:, 0].set(jnp.ravel(P, order='F'))
    I_n = jnp.identity(n)
    I_n2 = jnp.identity(n**2)
    for k in range(1, N_eta):
        big_mat = jnp.matmul(jnp.kron(I_n, P), I_n2 - jnp.diag(jnp.ravel(I_n, order='F')))
        vec = jnp.dot(big_mat, F_vecs[:, k-1])
        F_vecs = F_vecs.at[:, k].set(vec)
    F_vecs_sum = jnp.sum(F_vecs*jnp.log(F_vecs), axis=1)
    F_sum_mat = jnp.reshape(F_vecs_sum, (n, n), order='F')
    return -jnp.squeeze(jnp.dot(jnp.array(pi), jnp.diagonal(F_sum_mat)))

# pi must be a tuple
@functools.partial(jit, static_argnames=['pi', 'N_eta', 'alpha', 'use_abs_param'])
def loss_RTE_pi(Q, A, pi, N_eta, alpha, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    h_ret = compute_RTE_pi(P, pi, N_eta)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return h_ret - alpha*penalty

_comp_RTE_pi_grad = jacrev(loss_RTE_pi)
@functools.partial(jit, static_argnames=['pi', 'N_eta', 'alpha', 'use_abs_param'])
def comp_RTE_pi_grad(Q, A, pi, N_eta, alpha, use_abs_param=True):
    grad = _comp_RTE_pi_grad(Q, A, pi, N_eta, alpha, use_abs_param) 
    return grad

############################################################
# Weighted Return-Time Entropy formulation
############################################################
def precompute_weighted_RTE_pi(W, w_max, N_eta):
    n = jnp.shape(W)[0]
    D_idx = jnp.zeros((N_eta, n, n))
    for k in range(N_eta):
        for i in range(n):
            vec = jnp.where(W[i, :] > 0, k - W[i, :], -w_max)
            D_idx = D_idx.at[k, i].set(vec)
    return D_idx

@functools.partial(jit, static_argnames=['w_max', 'pi', 'N_eta'])
def compute_weighted_RTE_pi(P, D_idx, W, w_max, pi, N_eta):
    n = jnp.shape(P)[0]
    F_mats = jnp.zeros((N_eta + w_max, n, n))
    init_vals = (F_mats, P, D_idx, W, w_max)
    F_mats, _, _, _, _ = jax.lax.fori_loop(0, N_eta, weighted_FHTs_loop_body, init_vals)
    F_mats = F_mats[w_max:, :, :]
    F_sum_mat = jnp.sum(F_mats*jnp.log(jnp.where(F_mats == 0, 1, F_mats)), axis=0)
    return -(jnp.dot(jnp.array(pi), jnp.diagonal(F_sum_mat)))

@functools.partial(jit, static_argnames=['w_max', 'pi', 'N_eta', 'alpha', 'use_abs_param'])
def loss_weighted_RTE_pi(Q, A, D_idx, W, w_max, pi, N_eta, alpha, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    h_ret_w = compute_weighted_RTE_pi(P, D_idx, W, w_max, pi, N_eta)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return h_ret_w - alpha*penalty

_comp_weighted_RTE_pi_grad = jacrev(loss_weighted_RTE_pi)
@functools.partial(jit, static_argnames=['w_max', 'pi', 'N_eta', 'alpha', 'use_abs_param'])
def comp_weighted_RTE_pi_grad(Q, A, D_idx, W, w_max, pi, N_eta, alpha, use_abs_param=True):
    grad = _comp_weighted_RTE_pi_grad(Q, A, D_idx, W, w_max, pi, N_eta, alpha, use_abs_param) 
    return grad

############################################################
# Fastest Mixing Markov Chain formulation
############################################################
@jit
def compute_SLEM(P):
    eigs = jnp.linalg.eigvals(P)
    sorted_eigs = eigs[jnp.argsort(jnp.abs(eigs))]
    mu = jnp.squeeze(jnp.max(jnp.abs(sorted_eigs[:-1])))
    return jnp.real(mu)

@functools.partial(jit, static_argnames=['use_abs_param'])
def loss_FMMC(Q, A, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    mu = compute_SLEM(P)
    return mu

# Autodiff parametrized loss function
_comp_FMMC_grad = jacrev(loss_FMMC)
@functools.partial(jit, static_argnames=['use_abs_param'])
def comp_FMMC_grad(Q, A, use_abs_param=True):
    grad = _comp_FMMC_grad(Q, A, use_abs_param) 
    return grad

############################################################
# Auxiliary strategy analysis methods below
############################################################
# @jit
def compute_SPCPs(A, FHT_mats, node_pairs):
    """
    Compute shortest-path capture probabilities for given node pairs. 

    Parameters
    ----------
    FHT_mats : jaxlib.xla_extension.DeviceArray  
        Tensor of first-hitting time probability matrices. 
    node_pairs : jaxlib.xla_extension.DeviceArray  
        (D x 2) array whose rows contain ordered pairs of nodes, with robot locations  
        in first column and intruder locations in second column. 

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        (D x 3) array with each row containing an ordered node pair followed by
        the corresponding shortest-path capture probability in last column.
         
    
    See Also
    --------
    compute_FHT_probs
    graph_comp.get_shortest_path_distances
    """
    node_pair_SPDs = graph_comp.get_shortest_path_distances(A, node_pairs)
    SPCPs = jnp.full((jnp.shape(node_pairs)[0], 3), np.NaN)
    for k in range(jnp.shape(node_pairs)[0]):
        SPCPs = SPCPs.at[k, :2].set(node_pairs[k, :])
        SPCPs = SPCPs.at[k, 2].set(FHT_mats[node_pair_SPDs[k, 0], node_pair_SPDs[k, 1], node_pair_SPDs[k, 2] - 1])
    return SPCPs

@jit
def compute_diam_pair_cap_probs(F, diam_pairs):
    """
    Compute capture probabilities for all leaf node pairs separated by graph diameter. 

    Parameters
    ----------
    F : jaxlib.xla_extension.DeviceArray  
        Capture probability matrix. 
    diam_pairs : jaxlib.xla_extension.DeviceArray  
        Array of ordered pairs of leaf nodes separated by the graph diameter. 

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        (D x 3) array with each row containing an ordered diametric leaf node pair
        followed by the corresponding capture probability in the last column, 
        where D is the total number of ordered diametric pairs within the graph. 
    
    See Also
    --------
    compute_cap_probs
    graph_comp.get_diametric_pairs
    """
    dp_cap_probs = jnp.full((jnp.shape(diam_pairs)[0], 3), np.NaN)
    for k in range(jnp.shape(diam_pairs)[0]):
        dp_cap_probs = dp_cap_probs.at[k, :2].set(diam_pairs[k, :])
        dp_cap_probs = dp_cap_probs.at[k, 2].set(F[diam_pairs[k, 0], diam_pairs[k, 1]])
    return dp_cap_probs

@jit
def compute_diam_pair_CP_variance(F, diam_pairs):
    """
    Compute variance of capture probabilities for leaf node pairs separated by graph diameter. 

    Parameters
    ----------
    F : jaxlib.xla_extension.DeviceArray  
        Capture probability matrix. 
    diam_pairs : jaxlib.xla_extension.DeviceArray  
        Array of ordered pairs of leaf nodes separated by the graph diameter. 
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Variance of capture probabilities corresponding to diametric leaf node pairs. 
    
    See Also
    --------
    compute_diam_pair_cap_probs
    """
    dp_cap_probs = compute_diam_pair_cap_probs(F, diam_pairs)
    dp_CP_var = jnp.var(dp_cap_probs[:, 2])
    return dp_CP_var
