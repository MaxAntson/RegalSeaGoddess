import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)


def predict_on_dataset(model, X_test, threshold):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
    return y_preds, y_probs


def evaluate_model_performance(y_test, y_preds):
    weights = np.where(
        y_test == 1, 0.5 / (y_test == 1).sum(), 0.5 / (y_test == 0).sum()
    )
    precision = precision_score(y_test, y_preds, sample_weight=weights)
    recall = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds, sample_weight=weights)
    return precision, recall, f1
