# Script for defining new test specifications
import TestSpec as ts
import os
from StratCompJax import *

# test_spec_name = "Complete_Graph_Test_Specification_v2"

test_graphs = []
test_graph_names = {}
test_graph_codes = {}
test_taus = {}
test_start_time = time.time()

for i in range(1, 4):
    graph, graph_name = gen_split_star_G(2, i + 1, 2)
    graph_code = gen_graph_code(graph)
    test_graphs.append(graph)
    test_graph_names["test" + str(i)] = graph_name
    test_graph_codes["test" + str(i)] = graph_code
    test_taus["test" + str(i)] = graph_diam(graph)

for i in range(4, 7):
    graph, graph_name = gen_split_star_G(2, i - 2, 3)
    graph_code = gen_graph_code(graph)
    test_graphs.append(graph)
    test_graph_names["test" + str(i)] = graph_name
    test_graph_codes["test" + str(i)] = graph_code
    test_taus["test" + str(i)] = graph_diam(graph)

# graph, graph_name = gen_grid_G(3, 3)
# graph_code = gen_graph_code(graph)
# for i in range(3, 7):
#     test_graphs.append(graph)
#     test_graph_names["test" + str(i)] = graph_name
#     test_graph_codes["test" + str(i)] = graph_code


# tracked_vals = ["iters", "P_diff_sums", "P_diff_max_elts", "MCP_inds", "MCPs", "final_MCP", "final_iters"]

# num_tests = 3

# d_test_spec = ts.TestSpec(test_spec_name, num_tests, opt_params, schedules, tracked_vals, test_graph_names, test_taus, test_graph_codes)
d_test_spec= ts.TestSpec(test_spec_filepath=os.getcwd() + "/TestSpecs/splitstar_study_v1.json")
# d_test_spec.test_spec_name = test_spec_name
d_test_spec.graph_names = test_graph_names
d_test_spec.taus = test_taus
d_test_spec.graph_codes = test_graph_codes
d_test_spec.save_test_spec("splitstar_study_v1", os.getcwd() + "/TestSpecs")
