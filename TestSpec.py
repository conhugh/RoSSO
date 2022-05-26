# Test specification class definition:
import json
import os

class TestSpec:
    default_test_spec_filepath = os.getcwd() + "/TestSpecs/default_test_spec.json"

    def __init__(self, test_spec_filepath=None, test_spec_name=None, num_tests=None, optimizer_params=None, schedules=None, trackers=None, graph_names=None, taus=None, graph_codes=None):
        if test_spec_filepath != None:
            if os.path.exists(test_spec_filepath):
                # instantiate test_spec object from JSON file:
                with open(test_spec_filepath, "r") as test_spec_file:
                    json_string = test_spec_file.read()
                    test_spec_dict = json.loads(json_string)
                    self.test_spec_name = test_spec_dict["test_spec_name"]
                    self.num_tests = test_spec_dict["num_tests"]
                    self.optimizer_params = test_spec_dict["optimizer_params"]
                    self.schedules = test_spec_dict["schedules"]
                    self.trackers = test_spec_dict["trackers"]
                    self.graph_names = test_spec_dict["graph_names"]
                    self.taus = test_spec_dict["taus"]
                    self.graph_codes = test_spec_dict["graph_codes"]
            else:
                raise ValueError("Test specification file was not found at provided path.")
        elif all(arg is not None for arg in (test_spec_name, num_tests, optimizer_params, schedules, trackers, graph_names, taus, graph_codes)): 
            self.test_spec_name = test_spec_name
            self.num_tests = num_tests
            self.optimizer_params = optimizer_params
            self.schedules = schedules
            self.trackers = trackers
            self.graph_names = graph_names
            self.taus = taus
            self.graph_codes = graph_codes
        elif os.path.exists(TestSpec.default_test_spec_filepath):
            print("WARNING: If no test specification filepath is provided, values must be given for all other keyword arguments.")
            input("Press enter to initializing default test specification from " + TestSpec.default_test_spec_filepath + " ...")
            with open(TestSpec.default_test_spec_filepath, "r") as default_test_spec_file:
                json_string = default_test_spec_file.read()
                default_test_spec_dict = json.loads(json_string)
                self.test_spec_name = default_test_spec_dict["test_spec_name"]
                self.num_tests = default_test_spec_dict["num_tests"]
                self.optimizer_params = default_test_spec_dict["optimizer_params"]
                self.schedules = default_test_spec_dict["schedules"]
                self.trackers = default_test_spec_dict["trackers"]
                self.graph_names = default_test_spec_dict["graph_names"]
                self.taus = default_test_spec_dict["taus"]
                self.graph_codes = default_test_spec_dict["graph_codes"]
        else:
            raise ValueError("No test specification filepath provided, missing values for other keyword arguments, and could not find default test specification file at " + TestSpec.default_test_spec_filepath)
        
    def save_test_spec(self, test_spec_name, folder_path):
        filepath = os.path.join(folder_path, test_spec_name + ".json")
        print("Saving " + test_spec_name + " to " + filepath + " ... ")
        if os.path.exists(filepath):
            input("WARNING! A test specification file with this name already exists, press ENTER to continue and overwrite data.")
        json_string = json.dumps(self.__dict__, sort_keys=False, indent=4)
        with open(filepath, "w") as json_file:
            json_file.write(json_string)

    def validate_test_spec(self):
        required_fields = ["test_spec_name", "num_tests", "optimizer_params", "schedules", "trackers", "graph_names", "taus", "graph_codes"]
        missing_fields = []
        complete_fields = True
        for field in required_fields:
            if field not in self.__dict__.keys():
                complete_fields = False
                missing_fields.append(field)
        if not complete_fields:
            raise ValueError("Test specification is missing required fields: " + str(missing_fields))
        if len(self.graph_names) != len(self.graph_codes) or len(self.graph_names) != len(self.taus):
            raise ValueError("Test specification must have the same number of graph names, graph codes, and taus.")
        elif len(self.graph_names) != self.num_tests:
            raise ValueError("Test specification must have num_tests match the number of graph names, graph codes, and taus.")
        if self.optimizer_params["varying_optimizer_params"]:
            if any("test" + str(tnum + 1) not in self.optimizer_params.keys() for tnum in range(self.num_tests)):
                raise ValueError("If using varying optimizer params, must provide one set of params for each test, see varying_params_test_spec_example.json for required format.")
        if self.schedules["varying_schedules"]:
            if any("test" + str(tnum + 1) not in self.schedules.keys() for tnum in range(self.num_tests)):
                raise ValueError("If using varying schedules, must provide one set of schedules for each test, see varying_params_test_spec_example.json for required format.")
        
        

