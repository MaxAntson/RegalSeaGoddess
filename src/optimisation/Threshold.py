import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(model, X_val, y_val):
    """Use F1 Score weighted by class to find optimal threshold."""
    y_probs = model.predict_proba(X_val)[:, 1]
    weights = np.where(y_val == 1, 0.5 / (y_val == 1).sum(), 0.5 / (y_val == 0).sum())
    precision, recall, thresholds = precision_recall_curve(
        y_val, y_probs, sample_weight=weights
    )
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    best_threshold = thresholds[best_idx - 1] if best_idx > 0 else 0.5
    return best_threshold
