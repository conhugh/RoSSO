# Functionality related to analyzing, encoding, and decoding binary adjacency matrices for a variety of environment graphs 
import jax.numpy as jnp
import numpy as np
import graph_gen

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

def get_leaf_node_pairs(A):
    """
    Get the set of all leaf node pairs for the given environment graph.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        (D x 2) array with each row containing an ordered leaf node pair,
        where D is the total number of ordered leaf pairs within the graph. 
    
    """
    leaf_inds = jnp.sum(A, axis=1) == 2
    leaf_map = jnp.outer(leaf_inds, leaf_inds)
    leaf_node_pairs = jnp.argwhere(leaf_map)
    return leaf_node_pairs

def get_diametric_pairs(A):
    """
    Get the set of all leaf node pairs separated by the graph diameter
    for environment graph given by binary adjacency matrix `A`.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        (D x 2) array with each row containing an ordered diametric leaf node pair,
        where D is the total number of ordered diametric pairs within the graph. 
    
    """
    dg = graph_diam(A)
    diam_pairs = []
    A_ = A
    for k in range(2, dg):
        A_ = jnp.matmul(A, A_)
    diam_pairs = jnp.argwhere(A_ == 0)
    return diam_pairs

def get_shortest_path_distances(A, node_pairs):
    """
    Compute the shortest-path distance between given node pairs
    for environment graph given by binary adjacency matrix `A`.

    Considers distance from a node to itself to be 1, for consistency 
    with the robotic surveillance problem formulation. 
    For details, see https://arxiv.org/abs/2011.07604.

    Parameters
    ----------
    A : jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix of the environment graph.
    node_pairs : jaxlib.xla_extension.DeviceArray  
        (D x 2) array of pairs of nodes. 


    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        (D x 3) array with each row containing a node pair, with corresponding
        shortest-path distance in the third column. 
    """
    node_pair_SPDs = jnp.zeros((jnp.shape(node_pairs)[0], 3), dtype=int)
    node_pair_SPDs = node_pair_SPDs.at[:, :2].set(node_pairs)
    n = jnp.shape(A)[0]
    A_ = A
    for k in range(1, n):
        k_spd_pairs = jnp.argwhere(A_ != 0)
        for pair_ind in range(jnp.shape(node_pair_SPDs)[0]):
            if node_pair_SPDs[pair_ind, 2] == 0:
                if jnp.shape(jnp.argwhere((k_spd_pairs[:, 0] == node_pair_SPDs[pair_ind, 0]) * (k_spd_pairs[:, 1] == node_pair_SPDs[pair_ind, 1])))[0] != 0:
                    node_pair_SPDs = node_pair_SPDs.at[pair_ind, 2].set(k)
        A_ = jnp.matmul(A, A_)
    return node_pair_SPDs

def get_closest_sym_strat_grid(P_ref, P_comp, gridrows, gridcols, sym_index=None):
    """
    Find closest grid-graph strategy which is equivalent under grid symmetry.

    The grid-graph strategy 'P_comp' should be considered equivalent to other
    strategies or adjacency matrices wherein the graph's nodes are renumbered 
    such that the grid is transformed according to any of its symmetries. If 
    sym_index is not provided, this function generates the full set of 
    symmetry-equivalent strategies for 'P_comp', and returns the strategy which 
    most closely matches strategy 'P_ref'. If 'sym_index' is provided, rather 
    than computing the closest match to 'P_ref' among all symmetries of 'P_comp',
    only the specific symmetry of 'P_comp' corresponding to 'sym_index' is
    computed and returned.
    

    Parameters
    ----------
    P_ref : jaxlib.xla_extension.DeviceArray
        Strategy (transition probability matrix) for a grid graph.
    P_comp : jaxlib.xla_extension.DeviceArray
        Strategy (transition probability matrix) for a grid graph.
    gridrows : int
        The number of rows in the grid graph.
    gridcols : int
        The number of columns in the grid graph.
    sym_index : (int, optional)
        The index of the grid symmetry of 'P_comp' to compute.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy equivalent to P_comp. 
    int
        The index of the symmetry corresponding to the returned strategy.
    """
    if(gridrows == gridcols):
        P_syms = jnp.stack([P_comp, sq_grid_rot90(P_comp, gridrows, gridcols), grid_rot180(P_comp), 
                            sq_grid_rot270(P_comp, gridrows, gridcols), grid_row_reflect(P_comp, gridrows, gridcols), 
                            grid_col_reflect(P_comp, gridrows, gridcols), sq_grid_transpose(P_comp, gridrows, gridcols), 
                            sq_grid_antitranspose(P_comp, gridrows, gridcols)])
        sum_sq_diffs = jnp.full(8, np.Inf)
    elif(gridrows == 1 or gridcols == 1):
        P_syms = jnp.stack([P_comp, grid_rot180(P_comp)])
        sum_sq_diffs = jnp.full(2, np.Inf)
    else:
        P_syms = jnp.stack([P_comp, grid_rot180(P_comp), grid_row_reflect(P_comp, gridrows, gridcols), 
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
    """
    Generate equivalent grid-graph strategy with grid rows reflected.

    The grid-graph strategy (or binary adjacency matrix) 'M' should be
    considered equivalent to another strategy or adjacency matrix 
    wherein the graph's nodes are renumbered such that the grid is 
    "flipped about its horizontal axis". This function generates that 
    equivalent strategy or adjacency matrix.

    Parameters
    ----------
    M : jaxlib.xla_extension.DeviceArray
        Strategy or binary adjacency matrix for a grid graph.
    gridrows : int
        The number of rows in the grid graph.
    gridcols : int
        The number of columns in the grid graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy or binary adjacency matrix equivalent to M. 
    """
    grid_perm = jnp.flipud(jnp.identity(gridrows)) # antidiagonal permutation matrix
    block = jnp.identity(gridcols)
    M_perm = jnp.kron(grid_perm, block)
    M = jnp.matmul(M_perm, M)
    M = jnp.matmul(M, jnp.transpose(M_perm))
    return M

def grid_col_reflect(M, gridrows, gridcols):
    """
    Generate equivalent grid-graph strategy with grid columns reflected.

    The grid-graph strategy (or binary adjacency matrix) 'M' should be
    considered equivalent to another strategy or adjacency matrix 
    wherein the graph's nodes are renumbered such that the grid is 
    "flipped about its vertical axis". This function generates that 
    equivalent strategy or adjacency matrix.

    Parameters
    ----------
    M : jaxlib.xla_extension.DeviceArray
        Strategy or binary adjacency matrix for a grid graph.
    gridrows : int
        The number of rows in the grid graph.
    gridcols : int
        The number of columns in the grid graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy or binary adjacency matrix equivalent to M. 
    """
    grid_perm = jnp.flipud(jnp.identity(gridrows)) # antidiagonal permutation matrix
    block = jnp.identity(gridcols)
    M_perm = jnp.kron(block, grid_perm)
    M = jnp.matmul(jnp.transpose(M_perm), M)
    M = jnp.matmul(M, M_perm)
    return M

def grid_rot180(M):  
    """
    Generate equivalent grid-graph strategy with grid rotated 180 degrees.

    The grid-graph strategy (or binary adjacency matrix) 'M' should be
    considered equivalent to another strategy or adjacency matrix 
    wherein the graph's nodes are renumbered such that the grid is 
    "rotated 180 degrees about its center". This function generates 
    that equivalent strategy or adjacency matrix.

    Parameters
    ----------
    M : jaxlib.xla_extension.DeviceArray
        Strategy or binary adjacency matrix for a grid graph.

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy or binary adjacency matrix equivalent to M. 
    """
    M = jnp.fliplr(M)
    M = jnp.flipud(M)
    return M

def sq_grid_transpose(M, gridrows, gridcols):
    """
    Generate equivalent square-grid-graph strategy with grid transposed.

    For square grid graphs, the strategy (or binary adjacency matrix) 'M' 
    should be considered equivalent to another strategy or adjacency matrix 
    wherein the graph's nodes are renumbered such that the grid is transposed.
    This function generates that equivalent strategy or adjacency matrix.

    Parameters
    ----------
    M : jaxlib.xla_extension.DeviceArray
        Strategy or binary adjacency matrix for a grid graph.
    gridrows : int
        The number of rows in the grid graph.
    gridcols : int
        The number of columns in the grid graph.
        
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy or binary adjacency matrix equivalent to M. 
    """
    n = gridrows*gridcols
    M_reflect = jnp.full((n, n), np.NaN)
    identity_map = jnp.stack([jnp.outer(jnp.arange(n, dtype=int), jnp.ones(n, dtype=int)), 
                              jnp.outer(jnp.ones(n, dtype=int), jnp.arange(n, dtype=int))])
    node_map = jnp.flipud(jnp.arange(0, n).reshape(gridrows, gridcols))
    ref_map = jnp.flipud(jnp.transpose(node_map)).flatten()
    reflect_map = jnp.stack([jnp.outer(ref_map, jnp.ones(n, dtype=int)), 
                             jnp.outer(jnp.ones(n, dtype=int), ref_map)])
    M_reflect = M_reflect.at[identity_map[0, :, :], identity_map[1, :, :]].set(M[reflect_map[0, :, :], reflect_map[1, :, :]])
    return M_reflect

def sq_grid_antitranspose(M, gridrows, gridcols):
    """
    Generate equivalent square-grid-graph strategy with grid antitransposed.

    For square grid graphs, the strategy (or binary adjacency matrix) 'M' 
    should be considered equivalent to another strategy or adjacency matrix 
    wherein the graph's nodes are renumbered such that grid is antitransposed,
    that is, the grid is flipped about the anti-diagonal. This function 
    generates that equivalent strategy or adjacency matrix.

    Parameters
    ----------
    M : jaxlib.xla_extension.DeviceArray
        Strategy or binary adjacency matrix for a grid graph.
    gridrows : int
        The number of rows in the grid graph.
    gridcols : int
        The number of columns in the grid graph.
        
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy or binary adjacency matrix equivalent to M. 
    """
    M = sq_grid_transpose(M, gridrows, gridcols)
    M = grid_rot180(M)
    return M

def sq_grid_rot90(M, gridrows, gridcols):
    """
    Generate equivalent square-grid-graph strategy with grid rotated 90 degrees.

    For square grid graphs, the strategy (or binary adjacency matrix) 'M' 
    should be considered equivalent to another strategy or adjacency matrix 
    wherein the graph's nodes are renumbered such that grid is rotated 90 degrees
    about its center. This function generates that equivalent strategy or 
    adjacency matrix.

    Parameters
    ----------
    M : jaxlib.xla_extension.DeviceArray
        Strategy or binary adjacency matrix for a grid graph.
    gridrows : int
        The number of rows in the grid graph.
    gridcols : int
        The number of columns in the grid graph.
        
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy or binary adjacency matrix equivalent to M. 
    """
    M = sq_grid_transpose(M, gridrows, gridcols)
    M = grid_row_reflect(M, gridrows, gridcols)
    return M

def sq_grid_rot270(M, gridrows, gridcols):
    """
    Generate equivalent square-grid-graph strategy with grid rotated 270 degrees.

    For square grid graphs, the strategy (or binary adjacency matrix) 'M' 
    should be considered equivalent to another strategy or adjacency matrix 
    wherein the graph's nodes are renumbered such that grid is rotated 270 degrees
    about its center. This function generates that equivalent strategy or 
    adjacency matrix.

    Parameters
    ----------
    M : jaxlib.xla_extension.DeviceArray
        Strategy or binary adjacency matrix for a grid graph.
    gridrows : int
        The number of rows in the grid graph.
    gridcols : int
        The number of columns in the grid graph.
        
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Grid-graph strategy or binary adjacency matrix equivalent to M. 
    """
    M = grid_row_reflect(M, gridrows, gridcols)
    M = sq_grid_transpose(M, gridrows, gridcols)
    return M

if __name__ == '__main__':
    # A = graph_gen.gen_star_G(8)
    A = graph_gen.gen_line_G(10)
    print(A[0])
    print(gen_graph_code(A[0]))