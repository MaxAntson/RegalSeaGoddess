from training.Training import cross_validate

import numpy as np


def run_cross_validation_pipeline(dataset):
    precisions, recalls, f1_scores, best_thresholds, num_trees = cross_validate(dataset)
    metrics = {
        "cv": {
            "mean_f1": round(100 * float(np.mean(f1_scores)), 2),
            "std_f1": round(100 * float(np.std(f1_scores)), 2),
            "precision": [round(100 * p, 2) for p in precisions],
            "recall": [round(100 * r, 2) for r in recalls],
            "f1": [round(100 * f, 2) for f in f1_scores],
            "best_threshold": best_thresholds,
            "num_trees": num_trees,
        }
    }
    return metrics
