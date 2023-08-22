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

@functools.partial(jit, static_argnames=['use_abs_param'])
def comp_P_param(Q, A, use_abs_param=True):
    P = Q*A
    if use_abs_param:
        P = jnp.abs(P) # apply component-wise absolute-value
    else:
        P = jnp.maximum(jnp.zeros_like(P), P) # apply component-wise ReLU   
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize rows to generate valid prob dist 
    return P

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
def precompute_weighted_cap_probs(n, tau, W):
    indic_mat = jnp.zeros((n, n, n))
    for k in range(tau):
        indic_mat = indic_mat.at[k].set((k + 1)*jnp.ones((n, n)) == W)

    I = jnp.identity(n)
    E_ij = jnp.zeros((n, n, n**2, n**2))
    for i in range(n):
        for j in range(n):
            E_j = jnp.diag(jnp.ones(n) - I[:, j])
            E_ij = E_ij.at[i, j].set(jnp.kron(E_j, (jnp.outer(I[:, i], I[:, j]))))

    return indic_mat, E_ij

@functools.partial(jit, static_argnames=['w_max', 'tau'])
def compute_weighted_cap_probs(P, indic_mat, E_ij, W, w_max, tau):
    n = jnp.shape(P)[0]
    F_vecs = jnp.zeros((n**2, tau + w_max))

    for k in range(tau):
        P_direct = P*indic_mat[k]
        P_direct_vec = jnp.reshape(P_direct, n**2, order='F')

        multi_step_probs = jnp.zeros(n**2)
        for i in range(n):
            for j in range(n):
                multi_step_probs += P[i, j]*jnp.matmul(E_ij[i, j], F_vecs[:, k + w_max - W[i, j].astype(int)])

        F_vecs = F_vecs.at[:, k + w_max].set(P_direct_vec + multi_step_probs)

    F_vecs = F_vecs[:, w_max:]
    cap_probs = jnp.sum(F_vecs, axis=1)
    return cap_probs

@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs'])
def compute_weighted_LCPs(P, indic_mat, E_ij, W, w_max, tau, num_LCPs=1):
    cap_probs = compute_weighted_cap_probs(P, indic_mat, E_ij, W, w_max, tau)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        lcps = jnp.sort(cap_probs)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def loss_weighted_LCP(Q, A, indic_mat, E_ij, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_weighted_LCPs(P, indic_mat, E_ij, W, w_max, tau, num_LCPs)
    return jnp.mean(lcps)

# Autodiff parametrized loss function
_comp_weighted_LCP_grad = jacrev(loss_weighted_LCP)
@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def comp_avg_weighted_LCP_grad(Q, A, indic_mat, E_ij, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    grad = _comp_weighted_LCP_grad(Q, A, indic_mat, E_ij, W, w_max, tau, num_LCPs, use_abs_param)
    return grad

# pi must be a tuple
@functools.partial(jit, static_argnames=['w_max', 'tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def loss_weighted_LCP_pi(Q, A, indic_mat, E_ij, W, w_max, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_weighted_LCPs(P, indic_mat, E_ij, W, w_max, tau, num_LCPs)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return jnp.mean(lcps) - alpha*penalty

_comp_avg_weighted_LCP_pi_grad = jacrev(loss_weighted_LCP_pi)
@functools.partial(jit, static_argnames=['w_max', 'tau', 'pi', 'alpha', 'num_LCPs', 'use_abs_param'])
def comp_avg_weighted_LCP_pi_grad(Q, A, indic_mat, E_ij, W, w_max, tau, pi, alpha, num_LCPs=1, use_abs_param=True):
    grad = _comp_avg_weighted_LCP_pi_grad(Q, A, indic_mat, E_ij, W, w_max, tau, pi, alpha, num_LCPs, use_abs_param)
    return grad

############################################################
# Multi-Agent Stackelberg formulation
############################################################
def precompute_multi_cap_probs(n, N):
    combinations = list(itertools.product(range(1, n+1), repeat=N+1))
    return jnp.array(combinations)

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
def precompute_weighted_multi_cap_probs(n, N, tau, W):
    indic_mat = jnp.zeros((n, n, n))
    for k in range(tau):
        indic_mat = indic_mat.at[k].set((k + 1)*jnp.ones((n, n)) == W)

    I = jnp.identity(n)
    E_ij = jnp.zeros((n, n, n**2, n**2))
    for i in range(n):
        for j in range(n):
            E_j = jnp.diag(jnp.ones(n) - I[:, j])
            E_ij = E_ij.at[i, j].set(jnp.kron(E_j, (jnp.outer(I[:, i], I[:, j]))))
    
    combs = jnp.array(list(itertools.product(range(1, n+1), repeat=N+1)))
    return indic_mat, E_ij, combs

@functools.partial(jit, static_argnames=['w_max', 'tau'])
def compute_weighted_multi_cap_probs(Ps, indic_mat, E_ij, combs, W, w_max, tau):
    n = jnp.shape(Ps)[0]
    N = Ps.shape[2]
    F_vecs = jnp.zeros((n**2, tau + w_max, N))
    I = jnp.identity(n)

    for r in range(N):
        for k in range(tau):
            P_direct = Ps[:, :, r]*indic_mat[k]
            P_direct_vec = jnp.reshape(P_direct, n**2, order='F')

            multi_step_probs = jnp.zeros(n**2)
            for i in range(n):
                for j in range(n):
                    multi_step_probs = multi_step_probs + Ps[i, j, r]*jnp.matmul(E_ij[i, j], F_vecs[:, k + w_max - W[i, j].astype(int), r])
                    
            F_vecs = F_vecs.at[:, k + w_max, r].set(P_direct_vec + multi_step_probs)
    F_vecs = F_vecs[:, w_max:, :]
    indiv_cap_probs = jnp.reshape(jnp.sum(F_vecs, axis=1), (n, n, N), order='F')

    cap_probs = jnp.zeros(len(combs))
    for i in range(len(combs)):
        idx_vec = combs[i]
        not_cap_prob = jnp.prod(1 - indiv_cap_probs[idx_vec[:-1], idx_vec[-1], jnp.arange(N)])
        cap_probs = cap_probs.at[i].set(1 - not_cap_prob) 

    return jnp.reshape(cap_probs, (n**N, n), order='F')

@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs'])
def compute_weighted_multi_LCPs(Ps, W, w_max, tau, num_LCPs=1):
    cap_probs = compute_weighted_multi_cap_probs(Ps, W, w_max, tau)
    if num_LCPs == 1:
        lcps = jnp.min(cap_probs)
    elif num_LCPs > 1:
        lcps = jnp.sort(cap_probs)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def loss_weighted_multi_LCP(Qs, As, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    N = Qs.shape[2]
    Ps = jnp.zeros_like(Qs)
    for i in range(N):
        P = comp_P_param(Qs[:, :, i], As[:, :, i], use_abs_param)
        Ps = Ps.at[:, :, i].set(P)
    return jnp.mean(compute_weighted_multi_LCPs(Ps, W, w_max, tau, num_LCPs))

# Autodiff parametrized loss function
_comp_weighted_multi_LCP_grad = jacrev(loss_weighted_multi_LCP)
@functools.partial(jit, static_argnames=['w_max', 'tau', 'num_LCPs', 'use_abs_param'])
def comp_avg_weighted_multi_LCP_grad(Qs, As, W, w_max, tau, num_LCPs=1, use_abs_param=True):
    grad = _comp_weighted_multi_LCP_grad(Qs, As, W, w_max, tau, num_LCPs, use_abs_param)
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
    m = compute_MHT(P)
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
def compute_multi_MHT(Ps):
    n = Ps.shape[0]
    N = Ps.shape[2]
    big_I = jnp.identity(n**(N+1))

    mat = jnp.identity(n)
    for i in range(N):
        mat = jnp.kron(mat, Ps[:, :, i])

    combinations = list(itertools.product(range(1, n+1), repeat=N+1))
    jax_combinations = jnp.array(combinations)
    last_entry = jax_combinations[:, -1]
    other_entries = jax_combinations[:, :-1]
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
    # vectorized_cond() method required to avoid generation of nan values from 0*log(0) in jacrev gradient
    def vectorized_cond(pred, true_fun, false_fun, operand):
        true_op = jnp.where(pred, operand, 0)
        false_op = jnp.where(pred, 0, operand)
        return jnp.where(pred, true_fun(true_op), false_fun(false_op))
    
    entropy_rate_matrix = vectorized_cond(P > 0, lambda x: x*jnp.log(x), lambda x: 0.0, P)
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
@functools.partial(jit, static_argnames=['w_max', 'pi', 'N_eta'])
def compute_weighted_RTE_pi(P, W, w_max, pi, N_eta):
    n = jnp.shape(P)[0]
    F_vecs = jnp.zeros((n**2, N_eta + w_max))
    I = jnp.identity(n)
    for k in range(N_eta):
        indic_mat = ((k + 1)*jnp.ones((n, n)) == W)
        P_direct = P*indic_mat
        P_direct_vec = jnp.ravel(P_direct, order='F')

        multi_step_probs = jnp.zeros(n**2)
        for i in range(n):
            for j in range(n):
                E_j = jnp.diag(jnp.ones(n) - I[:, j])
                E_ij = jnp.kron(E_j, (jnp.outer(I[:, i], I[:, j])))
                multi_step_probs = multi_step_probs + P[i, j]*jnp.matmul(E_ij, F_vecs[:, k + w_max - W[i, j].astype(int)])

        F_vecs = F_vecs.at[:, k + w_max].set(P_direct_vec + multi_step_probs)

    F_vecs = F_vecs[:, w_max:]
    F_vecs_sum = jnp.sum(F_vecs*jnp.log(jnp.where(F_vecs == 0, 1, F_vecs)), axis=1)
    F_sum_mat = jnp.reshape(F_vecs_sum, (n, n), order='F')
    return -jnp.squeeze(jnp.dot(jnp.array(pi), jnp.diagonal(F_sum_mat)))

# pi must be a tuple
@functools.partial(jit, static_argnames=['w_max', 'pi', 'N_eta', 'alpha', 'use_abs_param'])
def loss_weighted_RTE_pi(Q, A, W, w_max, pi, N_eta, alpha, use_abs_param=True):
    n = len(pi)
    P = comp_P_param(Q, A, use_abs_param)
    h_ret_w = compute_weighted_RTE_pi(P, W, w_max, pi, N_eta)
    penalty = jnp.dot(jnp.dot(jnp.array(pi), P - jnp.identity(n)), jnp.dot(P.T - jnp.identity(n), jnp.array(pi))) # stationary distribution constraint
    return h_ret_w - alpha*penalty

_comp_weighted_RTE_pi_grad = jacrev(loss_weighted_RTE_pi)
@functools.partial(jit, static_argnames=['w_max', 'pi', 'N_eta', 'alpha', 'use_abs_param'])
def comp_weighted_RTE_pi_grad(Q, A, W, w_max, pi, N_eta, alpha, use_abs_param=True):
    grad = _comp_weighted_RTE_pi_grad(Q, A, W, w_max, pi, N_eta, alpha, use_abs_param) 
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
