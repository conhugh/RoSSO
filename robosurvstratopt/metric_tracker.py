# Class for defining metrics to be tracked during/after optimization process:
import os
# from inspect import signature
# from functools import wraps

import metric_definitions

class MetricTracker:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.metric_history = []
        self.evaluate_function = self._get_evaluate_function(metric_name)


    def _get_evaluate_function(self, metric_name):
        if metric_name in metric_definitions.METRICS_REGISTRY:
            eval_func =  metric_definitions.METRICS_REGISTRY[metric_name]
            return eval_func
        else:
            raise ValueError(f"Unknown metric name: {metric_name}. \
                               Ensure your metric has been defined in the 'METRIC_DEFINITIONS' dict in metric_evaluations.py")


    def evaluate(self, *args, **kwargs):
        return self.evaluate_function(*args, **kwargs)
    

    def update_history(self, *args, **kwargs):
        new_value = self.evaluate(*args, **kwargs)
        self.metric_history.append(new_value)


    def get_history(self):
        return self.metric_history


    def print_history(self, metrics_directory=None):
        history_str = '[' + ', '.join(map(str, self.metric_history)) + ']'
        if metrics_directory is not None:
            os.makedirs(metrics_directory, exist_ok=True) # ensure the directory exists
            file_path = os.path.join(metrics_directory, f"{self.metric_name}.txt")
            # Write history to the file, appending if the file already exists
            with open(file_path, 'a') as file:
                file.write(history_str + '\n')
        else:
            print(f"{self.metric_name} History: {history_str}")


    def print_final_value(self):
        print("Final " + str(self.metric_name) + " value: " + str(self.metric_history[-1]))