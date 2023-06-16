# Functionality related to generating, encoding, and decoding binary adjacency matrices 
# for a variety of environment graphs 
import jax
import jax.numpy as jnp
import numpy as np

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

def gen_rand_NUDEL_star_G(n, edge_len_UB, edge_len_LB=1):
    """
    Generate binary adjacency matrix for a star graph with `n` nodes.

    Parameters
    ----------
    n : int 
        Number of nodes in the star graph.
    edge_len_LB : int
        Lower bound on the travel time for edges in the graph (excl. self-loops).
    edge_len_UB : int
        Upper bound on the travel time for edges in the graph.    

    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the star graph with `n` nodes. 
    jaxlib.xla_extension.DeviceArray
        Weighted adjacency matrix for the star graph with `n` nodes. 
    int
        Largest travel time for any edge in the generated graph.
    String
        Description of the graph structure. 
    """
    if edge_len_LB < 1:
        raise ValueError("Lower bound on travel time cannot be less than 1.")
    if edge_len_LB > edge_len_UB:
        raise ValueError("Upper bound on travel time must be greater than or equal to the lower bound.")
    graph_name = "nudel_star_N" + str(n)
    A, _ = gen_star_G(n)
    seed = np.random.randint(1000)
    # seed = 1
    key = jax.random.PRNGKey(seed)
    W = jnp.identity(n)
    for i in range(1, n):
        key, subkey = jax.random.split(key)
        edge_len = int(edge_len_LB + jnp.round(jax.random.uniform(subkey)*(edge_len_UB - edge_len_LB)))
        W = W.at[0, i].set(edge_len)
        W = W.at[i, 0].set(edge_len)
    w_max = int(jnp.max(W))
    return A, W, w_max, graph_name

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
    String
        Description of the graph structure. 
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
    String
        Description of the graph structure. 
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
    
def gen_grid_G(grid_rows, grid_cols):
    """
    Generate binary adjacency matrix for a grid graph. 

    Parameters
    ----------
    grid_rows : int
        Number of rows of nodes in the grid graph.
    grid_cols : int 
        Number of columns of nodes in the grid graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the grid graph. 
    String
        Description of the graph structure. 
    """
    graph_name = "grid_R" + str(grid_rows) + "_C" + str(grid_cols)
    n = grid_cols*grid_rows
    A = jnp.identity(n)
    A = A + jnp.diag(jnp.ones(n - grid_rows), grid_rows)
    A = A + jnp.diag(jnp.ones(n - grid_rows), -grid_rows)
    for k in range(n):
        if k % grid_rows == 0:
            A = A.at[k, k + 1].set(1)
        elif k % grid_rows == (grid_rows - 1):
            A = A.at[k, k - 1].set(1)
        else:
            A = A.at[k, k + 1].set(1)
            A = A.at[k, k - 1].set(1)
    return A, graph_name

def gen_holy_grid_G(grid_rows, grid_cols, missing_nodes):
    """
    Generate binary adjacency matrix for a grid graph with holes. 

    Parameters
    ----------
    grid_rows : int
        Number of rows of nodes in the grid graph.
    grid_cols : int 
        Number of columns of nodes in the grid graph.
    missing_nodes : list of numpy.arrays
        List of zero-indexed (row, column) coordinates of nodes to remove. 
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the grid graph. 
    String
        Description of the graph structure. 
    """
    graph_name = "holygrid_R" + str(grid_rows) + "_C" + str(grid_cols)
    n = grid_cols*grid_rows
    A, _ = gen_grid_G(grid_cols, grid_rows)
    # get indices of rows/cols to remove from adjacency matrix:
    m_inds = np.full(len(missing_nodes), np.NaN, dtype=int)
    for i, m_node in enumerate(missing_nodes):
        m_inds[i] = int(m_node[1]*grid_rows + m_node[0])
    A_holy = jnp.delete(A, m_inds, axis=0)
    A_holy = jnp.delete(A_holy, m_inds, axis=1)
    return A_holy, graph_name


def gen_NUDEL_grid_G(width_vec, height_vec):
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
    String
        Description of the graph structure. 
    """
    width = len(width_vec) + 1
    height = len(height_vec) + 1
    A, _ = gen_grid_G(width, height)
    graph_name = "NUDEL_grid_W" + str(width) + "_H" + str(height)
    n = width*height
    W = jnp.identity(n)
    # W = W + jnp.diag(jnp.ones(n - height), height)
    # W = W + jnp.diag(jnp.ones(n - height), -height)
    width_diag = jnp.full([n - height], np.NaN)
    for i in range(width - 1):
        width_diag = width_diag.at[i*height:(i + 1)*height].set(width_vec[i]*np.ones(height))
    # for i in range(1, height):
    #     width_diag = np.concatenate([width_diag, width_vec])
    W = W + jnp.diag(width_diag, height)
    W = W + jnp.diag(width_diag, -height)
    for k in range(n):
        if k % height == 0:
            W = W.at[k, k + 1].set(height_vec[0])
        elif k % height == (height - 1):
            W = W.at[k, k - 1].set(height_vec[height - 2])
        else:
            W = W.at[k, k + 1].set(height_vec[k % height])
            W = W.at[k, k - 1].set(height_vec[(k % height) - 1])
    return A, W, graph_name


def gen_hallway_G(length, double_sided=True):
    """
    Generate binary adjacency matrix for a hallway graph. 

    Parameters
    ----------
    length : int 
        Number of nodes in along the center of the hallway graph.
    double_sided : bool
        Whether the hallway graph has leaf nodes on both sides. 
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the hallway graph. 
    String
        Description of the graph structure. 
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
    if double_sided:
        right_rooms = jnp.zeros((n, n)).at[:, :].set(jnp.diag(jnp.ones(length), -2*length)) \
                     + jnp.zeros((n, n)).at[:, :].set(jnp.diag(jnp.ones(length), 2*length))
        A = A + right_rooms

    return A, graph_name

def gen_rand_tree_G(n, req_depth):
    """
    Generate binary adjacency matrix for a random tree graph. 

    Parameters
    ----------
    n : int 
        Number of nodes in the tree graph.
    req_depth : int
        Required depth of the tree graph.
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the random tree graph. 
    String
        Description of the graph structure. 
    """
    if req_depth > n - 1:
        raise ValueError("Required maximum depth is too large for the number of nodes specified.")
    graph_name = "randomtree_D" + str(req_depth) + "_N" + str(n)
    A = jnp.identity(n)
    line_G, _ = gen_line_G(req_depth + 1)
    A = A.at[:(req_depth + 1), :(req_depth + 1)].set(line_G)
    d_nodes = []
    for i in range(req_depth + 1):
        d_nodes.append([i])
    nodes_rem = n - req_depth - 1
    curr_root = 0
    child_depth = 1
    curr_root_ind = d_nodes[child_depth - 1].index(curr_root)
    seed = np.random.randint(1000)
    key = jax.random.PRNGKey(seed)
    while nodes_rem != 0:
        num_new_children = 0
        curr_max_node = n - nodes_rem
        if child_depth == req_depth and curr_root_ind == len(d_nodes[child_depth - 1]):
            num_new_children = nodes_rem
        else:
            key, subkey = jax.random.split(key)
            num_new_children = int(jnp.ceil((jax.random.uniform(subkey))*(nodes_rem**(child_depth/(req_depth)))))
        if num_new_children != 0:
            child_nums = list(np.arange(n - nodes_rem, n - nodes_rem + num_new_children))
            d_nodes[child_depth].extend(child_nums)
            A = A.at[curr_root, curr_max_node:curr_max_node + num_new_children].set(jnp.ones(num_new_children))
            A = A.at[curr_max_node:curr_max_node + num_new_children, curr_root].set(jnp.ones(num_new_children))
            if curr_root == 0:
                A = A.at[0, 1].set(1)
                A = A.at[1, 0].set(1)
            A = A.at[curr_root, (n - nodes_rem)].set(1)
            A = A.at[(n - nodes_rem), curr_root].set(1)
        nodes_rem = nodes_rem - num_new_children
        new_root_ind = curr_root_ind + 1
        if new_root_ind == len(d_nodes[child_depth - 1]):
            child_depth = child_depth + 1
            curr_root = d_nodes[child_depth - 1][0]
            curr_root_ind = 0
        else:
            curr_root = d_nodes[child_depth - 1][new_root_ind]
            curr_root_ind = new_root_ind
    return A, graph_name

def gen_tree_G(n, tree_dict):
    """
    Generate binary adjacency matrix for the given tree graph. 

    Parameters
    ----------
    n : int 
        Number of nodes in the tree graph.
    tree_dict : dictionary
        Dictionary of integers specifying tree structure. [CREATE EXAMPLES FOR THIS]
    
    Returns
    -------
    jaxlib.xla_extension.DeviceArray
        Binary adjacency matrix for the tree graph described by tree_dict. 
    String
        Description of the graph structure. 
    """
    # EXAMPLE INPUT:
    # n = 5
    # tree_dict = {
    #     0 : [1], 
    #     1 : [2],
    #     2 : [3, 4],
    #     3 : None,
    #     4 : None
    # }
    graph_name = "tree_N" + str(n)
    # Check validity of tree_dict:
    keys = list(tree_dict.keys())
    if len(keys) != n:
        raise ValueError("Number of keys in dictionary does not match specified number of nodes in tree.")
    nodes = list(np.arange(n))
    if nodes != keys:
        raise ValueError("Every node in the tree must be a key in the dictionary, and every key in the dictionary must be a node in the tree.")
    # Generate binary adjacency matrix:
    A = jnp.identity(n)
    for key in keys:
        vals = tree_dict[key]
        if vals is not None:
            for val in vals:
                if val not in keys:
                    raise ValueError("Every value in the dictionary must be either a list of nodes in the tree or None.")
                else:
                    A = A.at[key, val].set(1)
                    A = A.at[val, key].set(1)
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
    String
        Description of the graph structure. 
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
    String
        Description of the graph structure. 
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
    String
        Description of the graph structure. 
    """
    graph_name = "completebipartite_L" + str(left_nodes) + "_R" + str(right_nodes)
    n = left_nodes + right_nodes
    A = jnp.identity(n)
    A = A.at[:left_nodes, left_nodes:].set(jnp.ones((left_nodes, right_nodes)))
    A = A.at[left_nodes:, :left_nodes].set(jnp.ones((right_nodes, left_nodes)))
    return A, graph_name