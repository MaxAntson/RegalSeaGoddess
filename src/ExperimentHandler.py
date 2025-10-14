import time
import os
import json


class ExperimentHandler:
    def __init__(self, experiment_title=""):
        experiment_name = time.strftime("%Y-%m-%d_%H-%M")
        if experiment_title is not None:
            experiment_name += "_" + experiment_title
        self.experiment_name = experiment_name

        outputs_path = "../outputs/experiments/"
        self.results_path = os.path.join(outputs_path, experiment_name)
        os.mkdir(self.results_path)
        print(f"Created experiment folder: {self.results_path}")

    def save_config(self, config):
        config_file = open(os.path.join(self.results_path, "config.json"), "w")
        json.dump(config, config_file, indent=4)
