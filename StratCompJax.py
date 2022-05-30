# Computation of quantities relevant to optimization of stochastic surveillance strategies
from unittest import TextTestResult
import numpy as np
import time
import jax
import functools
from jax import grad, jacrev, jit
import jax.numpy as jnp
# from StratViz import *

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

#     Returns
#     -------
#     String
#         Unique encoding of the binary adjacency matrix.
    
#     """
#     bin_string = "1"  # leading 1 added to avoid issues with decimal conversion in decoding
#     for i in range(A.shape[0] - 1):
#         for j in range(i + 1, A.shape[1]):
#             bin_string = bin_string + str(int(A[i, j]))
#     graph_num = int(bin_string, base=2)
#     graph_code = "N" + str(A.shape[0]) + "_" + str(graph_num)
#     return graph_code

def gen_graph_code(A):
    """
    Generate a unique code representing the environment graph.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray 
        Binary adjacency matrix of the environment graph.

    Returns
    -------
    String
        Unique encoding of the binary adjacency matrix.
    
    """
    bin_string = "1"  # leading 1 added to avoid issues with decimal conversion in decoding
    for i in range(jnp.shape(A)[0] - 1):
        for j in range(i + 1, jnp.shape(A)[1]):
            bin_string = bin_string + str(int(A[i, j]))
    graph_num = int(bin_string, base=2)
    graph_code = "N" + str(A.shape[0]) + "_" + str(graph_num)
    return graph_code


def gen_graph_hexcode(A):
    """
    Generate a unique hexadecimal code representing the environment graph.

    Parameters
    ----------
    A : numpy.ndarray 
        Binary adjacency matrix of the environment graph.

    Returns
    -------
    String
        Unique hexadecimal encoding of the binary adjacency matrix.

    """
    bin_string = "1" # leading 1 added to avoid issues with decimal conversion in decoding
    for i in range(A.shape[0] - 1):
        for j in range(i + 1, A.shape[1]):
            bin_string = bin_string + str(int(A[i, j]))
    graph_num = int(bin_string, base=2)
    graph_hex = hex(graph_num)
    graph_code = "N" + str(A.shape[0]) + "_" + str(graph_hex).lstrip("0x")
    return graph_code


def graph_decode(graph_code):
    """
    Generate binary adjacency matrix of environment graph from graph code.

    Parameters
    ----------
    graph_code : String
        Unique encoding of the binary adjacency matrix. 

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph.
    
    """
    n = int(graph_code[1:graph_code.find("_")])
    U_dec = int(graph_code[(graph_code.find("_") + 1):])
    U_bin = bin(U_dec)
    U_bin_list = list(U_bin[(U_bin.find("b") + 2):]) # skip leading 1 in binary string
    U_bin_list = [int(x) for x in U_bin_list] 
    inds = jnp.triu_indices(n, 1)
    U = jnp.zeros((n, n))
    U = U.at[inds].set(U_bin_list)
    A = U + jnp.transpose(U) + jnp.identity(n)
    return A

def graph_diam(A):
    """
    Compute diameter of graph described by binary adjacency matrix `A`.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph.

    Returns
    -------
    int
        Diameter of the environment graph.
    
    """
    n = jnp.shape(A)[0]
    if jnp.all(A != 0):
        return 1
    A_ = A
    for i in range(2, n):
        A_ = jnp.matmul(A, A_)
        if jnp.all(A_ != 0):
            return i
    return np.NaN


def gen_star_G(n):
    """
    Generate binary adjacency matrix for a star graph with `n` nodes.

    Parameters
    ----------
    n : int 
        Number of nodes in the star graph.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the star graph with `n` nodes. 
    """
    graph_name = "star_N" + str(n)
    A = jnp.identity(n)
    A = A.at[0, :].set(jnp.ones(n))
    A = A.at[:, 0].set(jnp.ones(n))
    return A, graph_name

def gen_line_G(n):
    """
    Generate binary adjacency matrix for a line graph with `n` nodes.

    Parameters
    ----------
    n : int 
        Number of nodes in the line graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the line graph with `n` nodes. 
    """
    graph_name = "line_N" + str(n)
    A = jnp.identity(n)
    A = A + jnp.diag(jnp.ones(n - 1), 1)
    A = A + jnp.diag(jnp.ones(n - 1), -1)
    return A, graph_name

def gen_split_star_G(left_leaves, right_leaves, num_line_nodes):
    """
    Generate binary adjacency matrix for a "split star" graph. 

    The "split star" graph has a line graph with `num_line_nodes` nodes, 
    with one end of the line being connected to an additional `left_leaves` 
    leaf nodes, and the other end having `right_leaves` leaf nodes. 

    Parameters
    ----------
    left_leaves : int 
        Number of leaf nodes on the left end of the line graph.
    right_leaves : int
        Number of leaf nodes on the right end of the line graph.
    num_line_nodes : int
        Number of nodes in the connecting line graph (excluding leaves).
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the split star graph. 
    """
    graph_name = "splitstar_L" + str(left_leaves) + "_R" + str(right_leaves) + "_M" + str(num_line_nodes)
    left_star = jnp.identity(left_leaves + 1)
    left_star = left_star.at[left_leaves, :].set(jnp.ones(left_leaves + 1))
    left_star = left_star.at[:, left_leaves].set(jnp.ones(left_leaves + 1))
    right_star, _ = gen_star_G(right_leaves + 1)
    mid_line, _ = gen_line_G(num_line_nodes)

    n = left_leaves + right_leaves + num_line_nodes
    split_star = jnp.identity(n)
    split_star = split_star.at[0:(left_leaves + 1), 0:(left_leaves + 1)].set(left_star)
    split_star = split_star.at[left_leaves:(left_leaves + num_line_nodes), left_leaves:(left_leaves + num_line_nodes)].set(mid_line)
    split_star = split_star.at[(left_leaves + num_line_nodes - 1):n, (left_leaves + num_line_nodes - 1):n].set(right_star)
    return split_star, graph_name
    
def gen_grid_G(width, height):
    """
    Generate binary adjacency matrix for a grid graph. 

    Parameters
    ----------
    width : int 
        Number of nodes in each row of the grid graph.
    height : int
        Number of nodes in each column of the grid graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the grid graph. 
    """
    graph_name = "grid_W" + str(width) + "_H" + str(height)
    n = width*height
    A = jnp.identity(n)
    A = A + jnp.diag(jnp.ones(n - height), height)
    A = A + jnp.diag(jnp.ones(n - height), -height)
    for k in range(n):
        if k % height == 0:
            A = A.at[k, k + 1].set(1)
        elif k % height == (height - 1):
            A = A.at[k, k - 1].set(1)
        else:
            A = A.at[k, k + 1].set(1)
            A = A.at[k, k - 1].set(1)
    return A, graph_name

def gen_cycle_G(n):
    """
    Generate binary adjacency matrix for a cycle graph with `n` nodes. 

    Parameters
    ----------
    n: int 
        Number of nodes in the cycle graph.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the cycle graph. 
    """
    graph_name = "cycle_N" + str(n)
    A, _ = gen_line_G(n)
    A = A.at[0, n - 1].set(1)
    A = A.at[n - 1, 0].set(1)
    return A, graph_name

def gen_complete_G(n):
    """
    Generate binary adjacency matrix for a complete graph with `n` nodes. 

    Parameters
    ----------
    n: int 
        Number of nodes in the complete graph.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the complete graph. 
    """
    graph_name = "complete_N" + str(n)
    A = jnp.ones((n, n))
    return A, graph_name

def gen_complete_bipartite_G(left_nodes, right_nodes):
    """
    Generate binary adjacency matrix for a complete bipartite graph. 

    Parameters
    ----------
    left_nodes: int 
        Number of nodes one one side of the the complete bipartite graph.
    right_nodes: int 
        Number of nodes one the other side of the the complete bipartite graph.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the complete bipartite graph. 
    """
    graph_name = "completebipartite_L" + str(left_nodes) + "_R" + str(right_nodes)
    n = left_nodes + right_nodes
    A = jnp.identity(n)
    A = A.at[:left_nodes, left_nodes:].set(jnp.ones((left_nodes, right_nodes)))
    A = A.at[left_nodes:, :left_nodes].set(jnp.ones((right_nodes, left_nodes)))
    return A

@functools.partial(jit, static_argnames=['tau'])
def compute_FHT_probs(P, F0, tau):
    """
    Compute First Hitting Time (FHT) Probability matrices.

    To compute the Capture Probability Matrix, we must sum the FHT
    Probability matrices from 1 up to `tau` time steps. 
    For more information see https://arxiv.org/pdf/2011.07604.pdf.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Array of `tau` distinct First Hitting Time Probability matrices. 
    """
    F0 = F0.at[:, :, 0].set(P)
    for i in range(1, tau):
        F0 = F0.at[:, :, i].set(jnp.matmul(P, (F0[:, :, i - 1] - jnp.diag(jnp.diag(F0[:, :, i - 1])))))
    return F0


@functools.partial(jit, static_argnames=['tau'])
def compute_cap_probs(P, F0, tau):
    """
    Compute Capture Probability Matrix.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix.
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices. 
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
        Capture Probability matrix. 
    
    See Also
    --------
    compute_FHT_probs
    """
    F = compute_FHT_probs(P, F0, tau)
    cap_probs = jnp.sum(F, axis=2)
    return cap_probs


@functools.partial(jit, static_argnames=['tau'])
def compute_MCP(P, F0, tau):
    """
    Compute Minimum Capture Probability.

    Parameters
    ----------
    P : numpy.ndarray 
        Transition probability matrix. 
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    
    Returns
    -------
    numpy.ndarray
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
    P : numpy.ndarray 
        Transition probability matrix. 
    F0 : numpy.ndarray 
        Placeholder to be populated with FHT Probability matrices.
    tau : int
        Intruder's attack duration. 
    num_LCPs : int
        Number of the lowest capture probabilities to compute. 
    
    Returns
    -------
    numpy.ndarray
        Set of `num_LCPs` lowest capture probabilities. 
    
    See Also
    --------
    compute_cap_probs
    """
    F = compute_cap_probs(P, F0, tau)
    F_vec = F.flatten('F')
    lcps = jnp.sort(F_vec)[0:num_LCPs]
    return lcps


# # closure which returns desired gradient computation function:
# def get_grad_func(num_LCPs=1, parametrization="ReLU", projection=None):
#     # validate input:
#     if parametrization is None and projection is None:
#         raise ValueError("Must specify either parametrization type or projection type.")
#     elif parametrization is not None and projection is not None:
#         raise ValueError("If specifying projection type, must set parametrization=None.")
#     # return desired gradient computation function:
#     if num_LCPs == 1 and parametrization is not None:
#         if parametrization=="ReLU":
#             return comp_MCP_grad_param
#         elif parametrization=="AbsVal":
#             return comp_MCP_grad_param_abs
#         else:
#             raise ValueError("Invalid parametrization type, must specify either \"ReLU\" or \"AbsVal\".")
#     elif num_LCPs != 1 and parametrization is not None:
#         if parametrization=="ReLU":
#             return comp_avg_LCP_grad_param
#         else:
#             raise ValueError("Invalid parametrization type, must specify either \"ReLU\" or \"AbsVal\".")
#     elif num_LCPs==1:  #implement different projection types here if desired
#         return comp_MCP_grad
#     else:
#         return comp_avg_LCP_grad

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

    # print("Devices available:")
    # print(jax.devices())










