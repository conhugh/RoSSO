# Computation of quantities relevant to optimization of stochastic surveillance strategies
import numpy as np
import time
import jax
import functools
from jax import grad, jacrev, jit
import jax.numpy as jnp

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

def gen_hallway_G(length, double_sided=True):
    """
    Generate binary adjacency matrix for a hallway graph. 

    Parameters
    ----------
    width : int 
        Number of nodes in each row of the hallway graph.
    height : int
        Number of nodes in each column of the hallway graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the grid graph. 
    """
    if double_sided:
        graph_name = "hallway2S_L" + str(length)
        n = 3*length
    else:
        graph_name = "hallway1S_L" + str(length)
        n = 2*length
    A = jnp.identity(n)
    hall_G, _ = gen_line_G(length)
    A = A.at[:length, :length].set(hall_G)
    left_rooms = jnp.zeros((n, n)).at[:2*length, :2*length].set(jnp.diag(jnp.ones(length), -length)) \
                + jnp.zeros((n, n)).at[:2*length, :2*length].set(jnp.diag(jnp.ones(length), length))
    A = A + left_rooms
    # left_rooms = left_rooms.at[:, length].set(jnp.diag(jnp.ones(length), -length))
    # A = A + jnp.diag(jnp.ones(length), -length)
    # A = A + jnp.diag(jnp.ones(length), -length)
    if double_sided:
        right_rooms = jnp.zeros((n, n)).at[:, :].set(jnp.diag(jnp.ones(length), -2*length)) \
                     + jnp.zeros((n, n)).at[:, :].set(jnp.diag(jnp.ones(length), 2*length))
        A = A + right_rooms
    # for k in range(n):

    #     if k % height == 0:
    #         A = A.at[k, k + 1].set(1)
    #     elif k % height == (height - 1):
    #         A = A.at[k, k - 1].set(1)
    #     else:
    #         A = A.at[k, k + 1].set(1)
    #         A = A.at[k, k - 1].set(1)
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
def compute_FHT_probs_vec(Pvec, tau):
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

# @functools.partial(jit, static_argnames=['tau'])
def compute_cap_probs_vec(Pvec, tau):
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
    # n = np.sqrt(jnp.shape(Pvec)[0])
    F = compute_FHT_probs_vec(Pvec, tau)
    cap_probs_vec = jnp.sum(F, axis=1)
    return cap_probs_vec

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

    # n = 5
    # P = jnp.reshape(jnp.arange(1, n**2 + 1,  dtype=float), (n, n), order='F')
    # print("P = ")
    # print(P)

    # P = np.zeros((n, n))
    # P[0, : ] = [0, 0.25, 0.25, 0.25, 0.25]
    # P[:, 0] = [0, 1, 1, 1, 1]
    # print("P = ")
    # print(P)

    # P = jnp.asarray(P)
    # tau = 2
    # F0 = jnp.full((n, n, tau), np.NaN)
    # F = compute_cap_probs(P, F0, tau)
    # print("F = ")
    # print(F)

    A, graph_name = gen_hallway_G(4)
    print(graph_name)
    print(A)

    # Pvec = P.flatten(order='F')

# ATTEMPT TO CHECK PL-INEQUALITY:



# BY-HAND JACOBIAN COMPUTATION: (NOT WORKING)
    # A = jnp.identity(n**2) - jnp.diag(jnp.identity(n).flatten(order='F'))
    # print("A = ")
    # print(A)

    # P_krons = jnp.kron(jnp.transpose(P), jnp.identity(n)) + jnp.kron(jnp.identity(n), P)

    # J = jnp.matmul(A, P_krons) + jnp.identity(n**2)
    # print("J by hand = ")
    # print(J)

# USING AUTODIFF FOR HESSIAN COMPUTATION:
    # tau = 2
    # # F0 = jnp.full((n, n, tau), np.NaN)
    # cap_probs_vec = compute_cap_probs_vec(Pvec, tau)

    # Agraph, _ = gen_star_G(n)
    # J1 = comp_CP_jac_nz(Pvec, tau)
    # print("J by autodiff = ")
    # print(J1)

    # Agraph, _ = gen_star_G(n)
    # Jz = comp_CP_jac(Pvec, Agraph, tau)
    # print("Jz by autodiff = ")
    # print(Jz)

    # kgrad = comp_CPk_grad(Pvec, Agraph, tau, 0)
    # print("grad of zeroeth cp: ")
    # print(kgrad)

    # kjelt = comp_CPk_grad_elt(Pvec, Agraph, tau, 0, 0)
    # print("zeroeth elt of grad of zeroeth cp: ")
    # print(kjelt)

    # khess_col = comp_CPk_hess_col(Pvec, Agraph, tau, 0, 0)
    # print("grad of zeroeth elt of grad of zeroeth cp: ")
    # print(khess_col)

    # khess = comp_CPk_hess(Pvec, Agraph, tau, 0)
    # print("hessian of zeroeth cp:")
    # print(khess)

    # npHess = np.asarray(khess)
    # eigvals = np.linalg.eigvals((1/2)*(npHess + np.transpose(npHess)))
    # max_eigval = np.max(eigvals)
    # min_eigval = np.min(eigvals)
    # print("(H + H')/2 max eigenvalue = " + str(max_eigval) + ", min eigenvalue = " + str(min_eigval))

    # B1 = jnp.kron(jnp.transpose(P), jnp.identity(n))
    # print("B1 = ")
    # print(B1)
    # B2 = jnp.kron(jnp.identity(n), P)
    # print("B2 = ")
    # print(B2)
    # permn2 = jnp.diag(jnp.ones(n**2 - n), k=-n) + jnp.diag(jnp.ones(n), n**2 - n)
    # # permn2 = permn2.at[5, n**2 - 1].set(1)
    # print("permn2 = ")
    # print(permn2)
    # permn = jnp.diag(jnp.ones(n - 1), k=-1) + jnp.diag(jnp.ones(5), n**2 - 5)
    # permn = permn.at[0, n- 1].set(1)
    # print("permn = ")
    # print(permn)


    # Hess_sum = jnp.zeros((n**2, n**2))
    # for k in range(1**2):    
    #     e_k = jnp.zeros((1, n**2))
    #     e_k = e_k.at[0, k].set(1)
    #     print("e_k = ")
    #     print(e_k)
    #     r_k = jnp.matmul(e_k, A)
    #     Hess = jnp.zeros((n**2, n**2))
    #     for j in range(n**2):
    #         e_j = jnp.zeros(n**2)
    #         e_j = e_j.at[j].set(1)
    #         M_j = jnp.reshape(e_j, (n, n), order='F')
    #         M_krons = jnp.kron(jnp.identity(n), M_j) + jnp.kron(jnp.transpose(M_j), jnp.identity(n))
    #         Hess_col_j = jnp.matmul(r_k, M_krons)
    #         Hess_col_j = jnp.reshape(Hess_col_j, n**2)
    #         Hess = Hess.at[:, j].set(Hess_col_j)
    #     Hess_sum = Hess_sum + Hess
    #     print("Hessian of cap prob " + str(k) + " = ")
    #     print(Hess)
    #     npHess = np.asarray(Hess)
    #     eigvals = np.linalg.eigvals((1/2)*(npHess + np.transpose(npHess)))
    #     max_eigval = np.max(eigvals)
    #     min_eigval = np.min(eigvals)
    #     print("(H + H')/2 max eigenvalue = " + str(max_eigval) + ", min eigenvalue = " + str(min_eigval))
    # print("Hess_sum = ")
    # print(Hess_sum)


    # test = jnp.outer(Pvec, jnp.ones(n**2))
    # print("test = ")
    # print(test)
    # # permtest = jnp.matmul(jnp.kron(jnp.identity(n), permn), Pvec)
    # permn2test = jnp.matmul(permn2, Pvec)
    # print("permn2test = ")
    # print(permn2test)
    # E = jnp.diag(jnp.identity(n).flatten(order='F'))
    # print("E = ")
    # print(E)
    # Evec = jnp.identity(n).flatten(order='F')
    # print("Evec = ")
    # print(Evec)
    # temp = jnp.matmul(jnp.kron(jnp.identity(n), jnp.ones((n, n))), E)
    # print("temp = ")
    # print(temp)
    # temptest = temp*test
    # print("temptest = ")
    # print(temptest)
    # Etest = jnp.matmul(Evec, test)
    # print("Etest = ")
    # print(Etest)
    # test2 = jnp.matmul(jnp.kron(jnp.ones((n, n)), jnp.identity(n)), test)
    # print("test2 = ")
    # print(test2)
    # B = B1 + B2
    # print("B = ")
    # print(B)
    # A = jnp.identity(n**2) - jnp.diag(jnp.identity(n).flatten(order='F'))
    # print("A = ")
    # print(A)
    # C = jnp.matmul(A, B)
    # print("C = ")
    # print(C)
    # D = C + jnp.identity(n**2)
    # print("D = ")
    # print(D)
    # k = 4
    # krow = D[k, :]
    # print("kth row = ")
    # print(krow)
    # print("Devices available:")
    # print(jax.devices())

    # test_opt_P = np.array([[0, 0.47, 0, 0.53, 0, 0, 0, 0, 0],
    #                        [0.30, 0, 0, 0, 0.70, 0, 0, 0, 0],
    #                        [0, 1.00, 0, 0, 0, 0, 0, 0, 0], 
    #                        [0.34, 0, 0, 0, 0.56, 0, 0.1, 0, 0],
    #                        [0, 0, 0, 0, 0, 0.60, 0, 0.40, 0],
    #                        [0, 0, 0.567, 0, 0, 0, 0, 0, 0.433],
    #                        [0, 0, 0, 1.00, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0.445, 0, 0.555, 0, 0],
    #                        [0, 0, 0, 0, 0, 0.30, 0, 0.70, 0]])








