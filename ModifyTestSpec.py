# Script for defining new test specifications
import TestSpec as ts
import os
from StratCompJax import *

test_spec_name = "Hallway_Graph_Test_Specification"

test_graphs = []
test_graph_names = {}
test_graph_codes = {}
test_taus = {}
test_start_time = time.time()
graph, graph_name = gen_hallway_G(4)
graph_code = gen_graph_code(graph)
test_graphs.append(graph)
test_graph_names["test1"] = graph_name
test_graph_codes["test1"] = graph_code
test_taus["test1"] = graph_diam(graph)
test_graph_names["test2"] = graph_name
test_graph_codes["test2"] = graph_code
test_taus["test2"] = graph_diam(graph) + 2
test_graph_names["test3"] = graph_name
test_graph_codes["test3"] = graph_code
test_taus["test3"] = graph_diam(graph) + 3

graph, graph_name = gen_hallway_G(4, double_sided=False)
graph_code = gen_graph_code(graph)
test_graphs.append(graph)
test_graph_names["test4"] = graph_name
test_graph_codes["test4"] = graph_code
test_taus["test4"] = graph_diam(graph)
test_graph_names["test5"] = graph_name
test_graph_codes["test5"] = graph_code
test_taus["test5"] = graph_diam(graph) + 2
test_graph_names["test6"] = graph_name
test_graph_codes["test6"] = graph_code
test_taus["test6"] = graph_diam(graph) + 3

# graph, graph_name = gen_grid_G(3, 3)
# graph_code = gen_graph_code(graph)
# for i in range(3, 7):
#     test_graphs.append(graph)
#     test_graph_names["test" + str(i)] = graph_name
#     test_graph_codes["test" + str(i)] = graph_code


# tracked_vals = ["iters", "P_diff_sums", "P_diff_max_elts", "MCP_inds", "MCPs", "final_MCP", "final_iters"]

# num_tests = 3

# d_test_spec = ts.TestSpec(test_spec_name, num_tests, opt_params, schedules, tracked_vals, test_graph_names, test_taus, test_graph_codes)
d_test_spec= ts.TestSpec(test_spec_filepath=os.getcwd() + "/TestSpecs/init_study_line_graph.json")
d_test_spec.test_spec_name = test_spec_name
d_test_spec.num_tests=6
d_test_spec.graph_names = test_graph_names
d_test_spec.taus = test_taus
d_test_spec.graph_codes = test_graph_codes
d_test_spec.save_test_spec("init_study_hallway_graph", os.getcwd() + "/TestSpecs")
