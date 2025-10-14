from evaluation.Prediction import predict_on_dataset, evaluate_model_performance
from training.Training import prepare_data_for_modelling

from ConfigHandler import config


def run_evaluation_pipeline(dataset, model):
    print("Evaluating model on test dataset")
    X_test, y_test = prepare_data_for_modelling(dataset)
    y_preds, _ = predict_on_dataset(model, X_test, config.PREDICTION.THRESHOLD)
    precision, recall, f1 = evaluate_model_performance(y_test, y_preds)
    metrics = {
        "test": {
            "precision": round(100 * precision, 2),
            "recall": round(100 * recall, 2),
            "f1": round(100 * f1, 2),
        }
    }
    return metrics
