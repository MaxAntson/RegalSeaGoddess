from sklearn.model_selection import train_test_split

from optimisation.Threshold import find_optimal_threshold
from training.Model import create_xgboost_model, train_xgboost_model
from training.Training import (
    prepare_data_for_modelling,
    calculate_class_weights,
)

from ConfigHandler import config


def run_training_pipeline(dataset):
    print("Training final model")
    X, y = prepare_data_for_modelling(dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.TRAINING.VAL_PROP,
        random_state=0,
        stratify=y,
    )
    pos_weight = calculate_class_weights(y_train)
    model = create_xgboost_model(pos_weight)
    model = train_xgboost_model(model, X_train, y_train, X_val, y_val)
    best_threshold = find_optimal_threshold(model, X_val, y_val)
    config.PREDICTION.THRESHOLD = best_threshold.item()
    return model
