import json
import os

import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ResultsHandler:
    def __init__(self, results_path):
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)
        self.results = {}

    def add_metric(self, name, value):
        self.results[name] = value

    def add_multiple_metrics(self, metrics):
        self.results.update(metrics)

    def save_results(self, results_name="results.json"):
        results_file_path = os.path.join(self.results_path, results_name)
        results_file = open(results_file_path, "w")
        json.dump(self.results, results_file, indent=4, cls=NpEncoder)
        print(f"Saved results to: {results_file_path}")
