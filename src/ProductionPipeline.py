import json
import os

from ConfigHandler import config


def run_production_pipeline(model, save_path):
    print("Saving the model and needed files for production")
    model_save_path = os.path.join(save_path, "model")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_model(os.path.join(model_save_path, "model.json"))

    with open(os.path.join(model_save_path, "feature_names.json"), "w") as f:
        json.dump(config.PREPROCESSING.ENVIRONMENT_DATA, f)
    with open(os.path.join(model_save_path, "threshold.txt"), "w") as f:
        f.write(str(config.PREDICTION.THRESHOLD))
