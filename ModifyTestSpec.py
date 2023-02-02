# Script for defining new test specifications
import os

from GraphGen import *
from StratCompJax import *
from StratViz import draw_env_graph
import TestSpec as ts

test_spec_name = "InitP250_Study_XD_Tree_Graphs_Test_Specification"

test_graphs = []
test_graph_names = {}
test_graph_codes = {}
test_taus = {}

node_nums = [7, 12]
tree_dicts = {
    "tree_dict1" : {
            0 : [1, 3, 4, 5], 
            1 : [2],
            2 : None,
            3 : None,
            4 : None,
            5 : [6],
            6 : None
        },
    "tree_dict2" : {
            0 : [1, 3, 4, 5, 7], 
            1 : [2],
            2 : None,
            3 : None,
            4 : None,
            5 : [6],
            6 : None,
            7 : [8],
            8 : [9, 10],
            9 : None,
            10 : [11],
            11 : None 
        }
}

# graph, graph_name = gen_line_G(3)
# graph_code = gen_graph_code(graph)
for i in range(2):
    graph, graph_name = gen_tree_G(node_nums[i], tree_dicts["tree_dict" + str(i + 1)])
    graph_code = gen_graph_code(graph)
    test_graphs.append(graph)
    draw_env_graph(graph, graph_name, os.getcwd())
    for t in range(3):
        test_graph_names["test" + str(i*3 + t + 1)] = graph_name
        test_graph_codes["test" + str(i*3 + t + 1)] = graph_code
        test_taus["test" + str(i*3 + t + 1)] = graph_diam(graph) + t



# graph, graph_name = gen_line_G(4)
# graph_code = gen_graph_code(graph)
# for i in range(5, 9):
#     test_graphs.append(graph)
#     test_graph_names["test" + str(i)] = graph_name
#     test_graph_codes["test" + str(i)] = graph_code
#     test_taus["test" + str(i)] = graph_diam(graph) + 3*(i - 5)

# graph, graph_name = gen_line_G(5)
# graph_code = gen_graph_code(graph)
# for i in range(9, 13):
#     test_graphs.append(graph)
#     test_graph_names["test" + str(i)] = graph_name
#     test_graph_codes["test" + str(i)] = graph_code
#     test_taus["test" + str(i)] = graph_diam(graph) + 3*(i - 9)


# graph, graph_name = gen_rand_tree_G(12, 3)
# graph_code = gen_graph_code(graph)
# test_graphs.append(graph)
# test_graph_names["test1"] = graph_name
# test_graph_codes["test1"] = graph_code
# test_taus["test1"] = graph_diam(graph)
# test_graph_names["test2"] = graph_name
# test_graph_codes["test2"] = graph_code
# test_taus["test2"] = graph_diam(graph) + 2
# test_graph_names["test3"] = graph_name
# test_graph_codes["test3"] = graph_code
# test_taus["test3"] = graph_diam(graph) + 3

# graph, graph_name = gen_rand_tree_G(12, 3)
# graph_code = gen_graph_code(graph)
# test_graphs.append(graph)
# test_graph_names["test4"] = graph_name
# test_graph_codes["test4"] = graph_code
# test_taus["test4"] = graph_diam(graph)
# test_graph_names["test5"] = graph_name
# test_graph_codes["test5"] = graph_code
# test_taus["test5"] = graph_diam(graph) + 2
# test_graph_names["test6"] = graph_name
# test_graph_codes["test6"] = graph_code
# test_taus["test6"] = graph_diam(graph) + 3

# graph, graph_name = gen_grid_G(3, 3)
# graph_code = gen_graph_code(graph)
# for i in range(3, 7):
#     test_graphs.append(graph)
#     test_graph_names["test" + str(i)] = graph_name
#     test_graph_codes["test" + str(i)] = graph_code


# tracked_vals = ["iters", "P_diff_sums", "P_diff_max_elts", "MCP_inds", "MCPs", "final_MCP", "final_iters"]

# num_tests = 3

# d_test_spec = ts.TestSpec(test_spec_name, num_tests, opt_params, schedules, tracked_vals, test_graph_names, test_taus, test_graph_codes)
d_test_spec= ts.TestSpec(test_spec_filepath=os.getcwd() + "/TestSpecs/init_study_tree_graphs.json")
d_test_spec.test_spec_name = test_spec_name
d_test_spec.num_tests = 6
d_test_spec.graph_names = test_graph_names
d_test_spec.taus = test_taus
d_test_spec.graph_codes = test_graph_codes
d_test_spec.save_test_spec("init_study_XD_tree_graphs", os.getcwd() + "/TestSpecs")
