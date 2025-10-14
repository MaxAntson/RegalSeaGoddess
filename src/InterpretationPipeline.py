from interpretation.Importance import get_feature_importance
from interpretation.Viz import get_partial_dependence, predict_over_accessible_area
import json


def run_interpretation_pipeline(model, dataset, accessible_area, save_path):
    print("Running interpretations...")
    feature_importance = get_feature_importance(model)
    with open(f"{save_path}/feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=4)
    get_partial_dependence(model, dataset, save_path)
    predict_over_accessible_area(model, dataset, accessible_area, save_path)
