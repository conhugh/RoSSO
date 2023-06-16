# Computation of quantities relevant to optimization of stochastic surveillance strategies
import functools

import jax
from jax import grad, jacrev, jit
import jax.numpy as jnp
import numpy as np

import GraphGen as gg

def init_rand_P(A):
    """
    Generate a random initial transition probability matrix.

    The robot's initial transition probability matrix must be row-stochastic 
    and consistent with the environment graph (described by `A`) to be valid. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Valid, random initial transition probability matrix. 
    """

    key = jax.random.PRNGKey(1)
    P0 = jnp.zeros_like(A, dtype='float32')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                P0 = P0.at[i, j].set(jax.random.uniform(key))
    P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0
    
def init_rand_P_key(A, key):
    """
    Generate a random initial transition probability matrix using PRNG key `key`.

    The robot's initial transition probability matrix must be row-stochastic 
    and consistent with the environment graph (described by `A`) to be valid. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray 
        Binary adjacency matrix of the environment graph.
    key : int 
        Jax PRNGKeyArray for random number generation.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Valid, random initial transition probability matrix. 
    """
    A_shape = jnp.shape(A)
    P0 = jax.random.uniform(key, A_shape)
    P0 = A*P0
    P0 = jnp.matmul(jnp.diag(1/jnp.sum(P0, axis=1)), P0)   # normalize to generate valid prob dist
    return P0

def init_rand_Ps(A, num):
    """
    Generate a set of `num` random initial transition probability matrices.

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
    
    See Also
    --------
    init_rand_P_key
    """
    key = jax.random.PRNGKey(0)
    initPs = jnp.zeros((jnp.shape(A)[0], jnp.shape(A)[1], num),  dtype='float32')
    for k in range(num):
        key, subkey = jax.random.split(key)
        initPs = initPs.at[:, : , k].set(init_rand_P_key(A, subkey))
    return initPs


@functools.partial(jit, static_argnames=['tau'])
def compute_FHT_probs(P, F0, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

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
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

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
def compute_NUDEL_FHT_probs_vec(P, W, w_max, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. This function
    computes these matrices in the case where the environment graph
    has integer (as opposed to unit) edge lengths. 
    For more information see https://arxiv.org/abs/1803.07705.

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
def compute_FHT_probs_vec(Pvec, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

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
    n = int(np.sqrt(len(Pvec)))
    F0 = jnp.full((len(Pvec), tau), np.NaN)
    F0 = F0.at[:, 0].set(Pvec)
    P = jnp.reshape(Pvec, (n, n), order='F')
    E = jnp.identity(n**2) - jnp.diag(jnp.identity(n).flatten(order='F'))
    for i in range(1, tau):
        F0 = F0.at[:, i].set(jnp.matmul(jnp.matmul(jnp.kron(jnp.identity(n), P), E), F0[:, i - 1]))
    return F0

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

# @functools.partial(jit, static_argnames=['tau'])
def compute_cap_probs_vec(Pvec, tau):  # MAY WANT TO REPLACE THIS FN BY SETTING AXIS=-1 IN SUM 
    """
    Compute Capture Probability Matrix.

    Parameters
    ----------
    P : jaxlib.xla_extension.DeviceArray 
        Transition probability matrix.
    F0 :  jaxlib.xla_extension.DeviceArray
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
    # n = np.sqrt(jnp.shape(Pvec)[0])
    F = compute_FHT_probs_vec(Pvec, tau)
    cap_probs_vec = jnp.sum(F, axis=1)
    return cap_probs_vec


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
    GraphGen.get_shortest_path_distances
    """
    node_pair_SPDs = gg.get_shortest_path_distances(A, node_pairs)
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
    GraphGen.get_diametric_pairs
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

@functools.partial(jit, static_argnames=['tau'])
def compute_MCP(P, F0, tau):
    """
    Compute Minimum Capture Probability.

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
        Minimum Capture Probability. 
    
    See Also
    --------
    compute_cap_probs
    """
    F = compute_cap_probs(P, F0, tau)
    mcp = jnp.min(F)
    return mcp
    
@functools.partial(jit, static_argnames=['tau', 'num_LCPs'])
def compute_LCPs(P, F0, tau, num_LCPs):
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
    F_vec = F.flatten('F')
    lcps = jnp.sort(F_vec)[0:num_LCPs]
    return lcps

# closure which returns desired gradient computation function:
def get_grad_func(grad_mode="MCP_parametrization"):
    if grad_mode == "MCP_parametrization":
        return comp_MCP_grad_param 
    elif grad_mode == "MCP_abs_parametrization":
        return comp_MCP_grad_param_abs
    elif grad_mode == "LCP_parametrization":
        return comp_avg_LCP_grad_param
    elif grad_mode == "MCP_projection":
        return comp_MCP_grad
    elif grad_mode == "LCP_projection":
        return comp_avg_LCP_grad
    else:
        raise ValueError("Invalid grad_mode specified!")

# Autodiff version of Min Cap Prob Gradient computation:
_comp_MCP_grad = jacrev(compute_MCP)
@functools.partial(jit, static_argnames=['tau'])
def comp_MCP_grad(P, A, F0, tau):
    G = _comp_MCP_grad(P, F0, tau)
    G = G*A
    return G

# Autodiff version of Lowest Cap Probs Gradient computation:
_comp_LCP_grads = jacrev(compute_LCPs)
@functools.partial(jit, static_argnames=['tau', 'num_LCPs'])
def comp_avg_LCP_grad(P, A, F0, tau, num_LCPs=None):
    G = _comp_LCP_grads(P, F0, tau, num_LCPs)
    G = G*A
    G_avg = jnp.mean(G, axis=0)
    return G_avg

# Parametrization of the P matrix
@jit
def comp_P_param(Q, A):
    P = Q*A
    P = jnp.maximum(jnp.zeros_like(P), P) # apply component-wise ReLU
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize rows to generate valid prob dist
    return P

# Parametrization of the P matrix
@jit
def comp_P_param_abs(Q, A):
    P = Q*A
    P = jnp.abs(P) # apply component-wise absolute-value
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize rows to generate valid prob dist
    return P

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau'])
def loss_MCP(Q, A, F0, tau):
    P = comp_P_param(Q, A)
    mcp = compute_MCP(P, F0, tau)
    return mcp

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau'])
def loss_MCP_abs(Q, A, F0, tau):
    P = comp_P_param_abs(Q, A)
    mcp = compute_MCP(P, F0, tau) 
    return mcp

# Autodiff parametrized loss function
_comp_MCP_grad_param = jacrev(loss_MCP)
@functools.partial(jit, static_argnames=['tau'])
def comp_MCP_grad_param(Q, A, F0, tau):
    grad = _comp_MCP_grad_param(Q, A, F0, tau) 
    return grad

# Autodiff parametrized loss function
_comp_MCP_grad_param_abs = jacrev(loss_MCP_abs)
@functools.partial(jit, static_argnames=['tau'])
def comp_MCP_grad_param_abs(Q, A, F0, tau):
    grad = _comp_MCP_grad_param_abs(Q, A, F0, tau) 
    return grad

# Loss function with constraints included in parametrization
@functools.partial(jit, static_argnames=['tau', 'num_LCPs'])
def loss_LCP(Q, A, F0, tau, num_LCPs=None):
    P = comp_P_param(Q, A)
    lcps = compute_LCPs(P, F0, tau, num_LCPs)
    return lcps

# Autodiff parametrized loss function
_comp_LCP_grads_param = jacrev(loss_LCP)
@functools.partial(jit, static_argnames=['tau', 'num_LCPs'])
def comp_avg_LCP_grad_param(Q, A, F0, tau, num_LCPs=None):
    J = _comp_LCP_grads_param(Q, A, F0, tau, num_LCPs) 
    grad = jnp.mean(J, axis=0)
    return grad

# TESTING -----------------------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)


