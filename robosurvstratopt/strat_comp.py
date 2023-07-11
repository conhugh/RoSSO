# Computation of quantities relevant to optimization of stochastic surveillance strategies
import functools

import jax
from jax import grad, jacrev, jit
import jax.numpy as jnp
import numpy as np

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
    if num == 1:
        initPs = jnp.squeeze(initPs)
    return initPs

@functools.partial(jit, static_argnames=['tau'])
def compute_FHT_probs(P, F0, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. 

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
        Array of `tau` distinct First Hitting Time Probability matrices. 
    """
    F0 = F0.at[:, :, 0].set(P)
    for i in range(1, tau):
        F0 = F0.at[:, :, i].set(jnp.matmul(P, (F0[:, :, i - 1] - jnp.diag(jnp.diag(F0[:, :, i - 1])))))
    return F0

@functools.partial(jit, static_argnames=['tau'])
def compute_FHT_probs_NF0(P, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. 

    Parameters
    ----------
    P : jaxlib.xla_extension.DeviceArray
        Transition probability matrix. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Array of `tau` distinct First Hitting Time Probability matrices. 
    """
    n = jnp.shape(P)[0]
    F0 = jnp.full((n, n, tau), np.NaN)
    F0 = F0.at[:, :, 0].set(P)
    for i in range(1, tau):
        F0 = F0.at[:, :, i].set(jnp.matmul(P, (F0[:, :, i - 1] - jnp.diag(jnp.diag(F0[:, :, i - 1])))))
    return F0

@functools.partial(jit, static_argnames=['w_max', 'tau']) 
def compute_weighted_FHT_probs_vec(P, W, w_max, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. This function
    computes these matrices in the case where the environment graph
    has integer (as opposed to unit) edge lengths. 

    Parameters
    ----------
    P : jaxlib.xla_extension.DeviceArray
        Transition probability matrix. 
    W : jaxlib.xla_extension.DeviceArray
        Integer-valued travel time matrix. 
    F0 : jaxlib.xla_extension.DeviceArray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Array of `tau` distinct First Hitting Time Probability matrices. 
    """
    n = jnp.shape(P)[0]
    F_vecs = jnp.zeros((n**2, tau + w_max))
    I = jnp.identity(n)

    for k in range(tau):
        indic_mat = ((k + 1)*jnp.ones((n, n)) == W)
        P_direct = P*indic_mat
        P_direct_vec = jnp.reshape(P_direct, n**2, order='F')

        multi_step_probs = jnp.zeros(n**2)
        for i in range(n):
            for j in range(n):
                E_j = jnp.diag(jnp.ones(n) - I[:, j])
                E_ij = jnp.kron(E_j, (jnp.outer(I[:, i], I[:, j])))
                multi_step_probs = multi_step_probs + P[i, j]*jnp.matmul(E_ij, F_vecs[:, k + w_max - W[i, j].astype(int)])
                
        F_vecs = F_vecs.at[:, k + w_max].set(P_direct_vec + multi_step_probs)
        
    F_v = F_vecs[:, w_max:]
    return F_v

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
    F = compute_FHT_probs(P, F0, tau)
    cap_probs = jnp.sum(F, axis=2)
    return cap_probs

@functools.partial(jit, static_argnames=['tau'])
def compute_cap_probs_NF0(P, tau):
    """
    Compute Capture Probability Matrix.

    Parameters
    ----------
    P : jaxlib.xla_extension.DeviceArray 
        Transition probability matrix.
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
    F = compute_FHT_probs_NF0(P, tau)
    cap_probs = jnp.sum(F, axis=2)
    return cap_probs
    
# Parametrization of the P matrix
@functools.partial(jit, static_argnames=['use_abs_param'])
def comp_P_param(Q, A, use_abs_param=True):
    P = Q*A
    if use_abs_param:
        P = jnp.abs(P) # apply component-wise absolute-value
    else:
        P = jnp.maximum(jnp.zeros_like(P), P) # apply component-wise ReLU   
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize rows to generate valid prob dist 
    return P

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
    F = compute_cap_probs(P, F0, tau)
    if num_LCPs == 1:
        lcps = jnp.min(F)
    elif num_LCPs > 1:
        F_vec = F.flatten('F')
        lcps = jnp.sort(F_vec)[0:num_LCPs]
    else:
        raise ValueError("Invalid num_LCPs specified!")
    return lcps

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau', 'num_LCPs', 'use_abs_param'])
def loss_LCP(Q, A, F0, tau, num_LCPs=1, use_abs_param=True):
    P = comp_P_param(Q, A, use_abs_param)
    lcps = compute_LCPs(P, F0, tau, num_LCPs)
    return lcps

# Autodiff parametrized loss function
_comp_LCP_grads = jacrev(loss_LCP)
@functools.partial(jit, static_argnames=['tau', 'num_LCPs', 'use_abs_param'])
def comp_avg_LCP_grad(Q, A, F0, tau, num_LCPs=1, use_abs_param=True):
    J = _comp_LCP_grads(Q, A, F0, tau, num_LCPs, use_abs_param) 
    grad = jnp.mean(J, axis=0)
    return grad

############################################################
# Auxiliary strategy analysis functions below
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
