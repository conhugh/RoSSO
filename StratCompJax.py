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
    A : numpy.ndarray 
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
    A : numpy.ndarray 
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
    initPs = jnp.zeros((A.shape[0], A.shape[1], num),  dtype='float32')
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

_CP_jac = jacrev(compute_cap_probs_vec)
def comp_CP_jac(Pvec, A, tau):
    G = _CP_jac(Pvec, tau)
    A_id = jnp.diag(A.flatten(order='F'))
    G = jnp.matmul(G, A_id)
    return G

def comp_CPk_grad(Pvec, A, tau, k):
    G = comp_CP_jac(Pvec, A, tau)
    kgrad = G[k, :]
    return kgrad

def comp_CPk_grad_elt(Pvec, A, tau, k, j):
    kgrad = comp_CPk_grad(Pvec, A, tau, k)
    kj_elt = kgrad[j]
    return kj_elt

_CPk_hess_col = jacrev(comp_CPk_grad_elt)
def comp_CPk_hess_col(Pvec, A, tau, k, j):
    Hcol = _CPk_hess_col(Pvec, A, tau, k, j)
    return Hcol

def comp_CPk_hess(Pvec, A, tau, k):
    H = jnp.full((len(Pvec), len(Pvec)), np.NaN)
    for j in range(len(Pvec)):
        H = H.at[:, j].set(comp_CPk_hess_col(Pvec, A, tau, k, j))
    return H

def comp_CP_jac_nz(Pvec, tau):
    G = _CP_jac(Pvec, tau)
    return G

# Autodiff version of Min Cap Prob Gradient computation:
_comp_MCP_grad = jacrev(compute_MCP)
# wrapper function:
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
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize to generate valid prob dist
    return P

# Parametrization of the P matrix
@jit
def comp_P_param_abs(Q, A):
    P = Q*A
    P = jnp.abs(P) # apply component-wise absolute-value
    P = jnp.matmul(jnp.diag(1/jnp.sum(P, axis=1)), P)   # normalize to generate valid prob dist
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

# # Autodiff parametrized loss function
# _comp_MCP_grad_param_test = grad(loss_MCP)
# @functools.partial(jit, static_argnames=['tau'])
# def comp_MCP_grad_param_test(Q, A, F0, tau):
#     grad = _comp_MCP_grad_param_test(Q, A, F0, tau) 
#     return grad

# @functools.partial(jit, static_argnames=['tau'])
# def comp_MCP_grad_param_extra(Q, A, F0, tau, num_LCPs=None):
#     grad = _comp_MCP_grad_param(Q, A, F0, tau) 
#     return grad

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

@jit
def proj_onto_simplex(P):
    """
    Project rows of the Transition Probability Matrix `P` onto the probability simplex.

    To ensure gradient-based updates to the Transition Probability Matrix maintain
    row-stochasticity, the rows of the updated Transition Probability Matrix are projected 
    onto the nearest point on the probability n-simplex, where `n` is the number of
    columns of `P`.  For further explanation, see [LINK TO DOCUMENT ON GITHUB], and 
    for more about the projection algorithm used, see https://arxiv.org/abs/1309.1541.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition Probability Matrix after gradient update, potentially invalid. 
    
    Returns
    -------
    numpy.ndarray
        Valid Transition Probability Matrix nearest to `P` in Euclidian sense. 
    """
    n = P.shape[0]
    sort_map = jnp.fliplr(jnp.argsort(P, axis=1))
    X = jnp.full_like(P, np.nan)
    for i  in range (n):
        for j in range (n):
            X = X.at[i, j].set(P[i, sort_map[i, j]])
    X_tmp = jnp.matmul(jnp.cumsum(X, axis=1) - 1, jnp.diag(1/jnp.arange(1, n + 1)))
    rho_vals = jnp.sum(X > X_tmp, axis=1) - 1
    lambda_vals = -X_tmp[jnp.arange(n), rho_vals]
    X_new = jnp.maximum(X + jnp.outer(lambda_vals, jnp.ones(n)), jnp.zeros([n, n]))
    P_new = jnp.full_like(P, np.nan)
    for i in range(n):
        for j in range(n):
            P_new = P_new.at[i, sort_map[i, j]].set(X_new[i, j])
    return P_new

@jit
def proj_onto_simplex_large(P):
    """
    Project rows of the Transition Probability Matrix `P` onto the probability simplex.

    To ensure gradient-based updates to the Transition Probability Matrix maintain
    row-stochasticity, the rows of the updated Transition Probability Matrix are projected 
    onto the nearest point on the probability n-simplex, where `n` is the number of
    columns of `P`.  For further explanation, see [LINK TO DOCUMENT ON GITHUB], and 
    for more about the projection algorithm used, see https://arxiv.org/abs/1309.1541.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition Probability Matrix after gradient update, potentially invalid. 
    
    Returns
    -------
    numpy.ndarray
        Valid Transition Probability Matrix nearest to `P` in Euclidian sense. 
    """
    n = P.shape[0]
    P_new = jnp.full_like(P, np.nan)
    for i in range(n):
        P_new = P_new.at[i, :].set(proj_row_onto_simplex(P[i, :]))
    return P_new

@jit
def proj_row_onto_simplex(row):
    """
    Project rows of the Transition Probability Matrix `P` onto the probability simplex.

    To ensure gradient-based updates to the Transition Probability Matrix maintain
    row-stochasticity, the rows of the updated Transition Probability Matrix are projected 
    onto the nearest point on the probability n-simplex, where `n` is the number of
    columns of `P`.  For further explanation, see [LINK TO DOCUMENT ON GITHUB], and 
    for more about the projection algorithm used, see https://arxiv.org/abs/1309.1541.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition Probability Matrix after gradient update, potentially invalid. 
    
    Returns
    -------
    numpy.ndarray
        Valid Transition Probability Matrix nearest to `P` in Euclidian sense. 
    """
    n = len(row)
    sort_map = jnp.flip(jnp.argsort(row))
    X = jnp.full_like(row, np.nan)
    for j in range (n):
        X = X.at[j].set(row[sort_map[j]])
    X_tmp = jnp.matmul(jnp.cumsum(X) - 1, jnp.diag(1/jnp.arange(1, n + 1)))
    rho = jnp.sum(X > X_tmp) - 1
    lambda_ = -X_tmp[rho]
    X_new = jnp.maximum(X + lambda_, jnp.zeros((n)))
    new_row = jnp.full_like(row, np.nan)
    for j in range(n):
        new_row = new_row.at[sort_map[j]].set(X_new[j])
    return new_row

def proj_row_onto_simplex_test(row):
    """
    Project rows of the Transition Probability Matrix `P` onto the probability simplex.

    To ensure gradient-based updates to the Transition Probability Matrix maintain
    row-stochasticity, the rows of the updated Transition Probability Matrix are projected 
    onto the nearest point on the probability n-simplex, where `n` is the number of
    columns of `P`.  For further explanation, see [LINK TO DOCUMENT ON GITHUB], and 
    for more about the projection algorithm used, see http://www.optimization-online.org/DB_FILE/2014/08/4498.pdf.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition Probability Matrix after gradient update, potentially invalid. 
    
    Returns
    -------
    numpy.ndarray
        Valid Transition Probability Matrix nearest to `P` in Euclidian sense. 
    """
    v = []
    vt = []
    v.append(row[0])
    rho = row[0] - 1
    # FIRST PASS:
    for k in range(1, len(row)):
        yn = row[k]
        if yn > rho:
            rho = rho + (yn - rho)/(len(v) + 1)
            if rho > (yn - 1):
                v.append(yn)
            else:
                vt.append(v)
                v.clear()
                v.append(yn)
                rho = yn - 1
    # CLEANUP PASS:
    for k in range(len(vt)):
        yn = vt[k]
        if yn > rho:
            v.append(yn)
            rho = rho + (yn - rho)/(len(v))
    # ELEMENT ELIMINATION LOOP:
    len_change = True
    while len_change:
        v_len = len(v)
        k = 0
        while k < len(v):
            if v[k] <= rho:
                y = v.pop(k)
                rho = rho + (rho - y)/(len(v))
            else:
                k = k + 1
        if v_len == len(v):
            len_change = False
    tau = rho
    # PROJECTION:
    row = jnp.maximum(row - tau, jnp.zeros(jnp.shape(row)))
    return row

def get_closest_sym_strat_grid(P_ref, P_comp, gridrows, gridcols, sym_index=None):
    if(gridrows == gridcols):
        P_syms = jnp.stack([P_comp, sq_grid_rot90(P_comp, gridrows, gridcols), grid_rot180(P_comp), \
                sq_grid_rot270(P_comp, gridrows, gridcols), grid_row_reflect(P_comp, gridrows, gridcols), \
                grid_col_reflect(P_comp, gridrows, gridcols), sq_grid_transpose(P_comp, gridrows, gridcols), \
                sq_grid_antitranspose(P_comp, gridrows, gridcols)
                ])
        sum_sq_diffs = jnp.full(8, np.Inf)
    elif(gridrows == 1 or gridcols == 1):
        P_syms = jnp.stack([P_comp, grid_rot180(P_comp)])
        sum_sq_diffs = jnp.full(2, np.Inf)
    else:
        P_syms = jnp.stack([P_comp, grid_rot180(P_comp), grid_row_reflect(P_comp, gridrows, gridcols), \
                                grid_col_reflect(P_comp, gridrows, gridcols)])
        sum_sq_diffs = jnp.full(4, np.Inf)

    if sym_index is not None:
        P_closest = jnp.reshape(P_syms[sym_index, :, :], (gridrows*gridcols, gridrows*gridcols))
    else:
        for k in range(jnp.shape(P_syms)[0]):
            sum_sq_diffs = sum_sq_diffs.at[k].set(jnp.sum((P_syms[k, :, :] - P_ref)**2))
        sym_index = jnp.argwhere(sum_sq_diffs == jnp.min(sum_sq_diffs))
        P_closest = jnp.reshape(P_syms[sym_index, :, :], (gridrows*gridcols, gridrows*gridcols))
    return P_closest, sym_index

def grid_row_reflect(M, gridrows, gridcols):
    grid_perm = jnp.flipud(jnp.identity(gridrows)) # antidiagonal permutation matrix
    block = jnp.identity(gridcols)
    M_perm = jnp.kron(grid_perm, block)
    M = jnp.matmul(M_perm, M)
    M = jnp.matmul(M, jnp.transpose(M_perm))
    return M

def grid_col_reflect(M, gridrows, gridcols):
    grid_perm = jnp.flipud(jnp.identity(gridrows)) # antidiagonal permutation matrix
    block = jnp.identity(gridcols)
    M_perm = jnp.kron(block, grid_perm)
    M = jnp.matmul(jnp.transpose(M_perm), M)
    M = jnp.matmul(M, M_perm)
    return M

def grid_rot180(M):  
    M = jnp.fliplr(M)
    M = jnp.flipud(M)
    return M

def sq_grid_transpose(M, gridrows, gridcols):
    n = gridrows*gridcols
    M_reflect = jnp.full((n, n), np.NaN)
    identity_map = jnp.stack([jnp.outer(jnp.arange(n, dtype=int), jnp.ones(n, dtype=int)), \
                                jnp.outer(jnp.ones(n, dtype=int), jnp.arange(n, dtype=int))])
    node_map = jnp.flipud(jnp.arange(0, n).reshape(gridrows, gridcols))
    ref_map = jnp.flipud(jnp.transpose(node_map)).flatten()
    reflect_map = jnp.stack([jnp.outer(ref_map, jnp.ones(n, dtype=int)), \
                                jnp.outer(jnp.ones(n, dtype=int), ref_map)])
    M_reflect = M_reflect.at[identity_map[0, :, :], identity_map[1, :, :]].set(M[reflect_map[0, :, :], reflect_map[1, :, :]])
    return M_reflect

def sq_grid_antitranspose(M, gridrows, gridcols):
    M = sq_grid_transpose(M, gridrows, gridcols)
    M = grid_rot180(M)
    return M

def sq_grid_rot90(M, gridrows, gridcols):
    M = sq_grid_transpose(M, gridrows, gridcols)
    M = grid_row_reflect(M, gridrows, gridcols)
    return M

def sq_grid_rot270(M, gridrows, gridcols):
    M = grid_row_reflect(M, gridrows, gridcols)
    M = sq_grid_transpose(M, gridrows, gridcols)
    return M



# TESTING -----------------------------------------------------------------------------------------
if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)


