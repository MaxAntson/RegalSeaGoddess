import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split

from optimisation.Threshold import find_optimal_threshold
from training.Model import create_xgboost_model, train_xgboost_model
from evaluation.Prediction import predict_on_dataset, evaluate_model_performance

from ConfigHandler import config


def prepare_data_for_modelling(dataset):
    X = dataset[config.PREPROCESSING.ENVIRONMENT_DATA].values
    if "label" in dataset:
        y = dataset["label"].values
    else:
        y = None
    return X, y


def create_spatial_folds(dataset):
    """TODO: improve distance calculation to account for spherical geometry"""
    dataset = dataset.to_crs(epsg=3857)
    coords = np.vstack([dataset.geometry.x, dataset.geometry.y]).T
    tree = cKDTree(coords)
    fold_ids = -1 * np.ones(len(dataset), dtype=int)

    current_fold = 0
    for i in range(len(dataset)):
        if fold_ids[i] != -1:
            continue
        fold_ids[i] = current_fold
        neighbors = tree.query_ball_point(
            coords[i], r=config.TRAINING.MIN_DISTANCE_BETWEEN_FOLDS_M
        )
        neighbors.remove(i)
        fold_ids[neighbors] = current_fold
        # Cycle through - i.e. each fold will be distributed across the map.
        # Good balance between aiming for both spatial separation and decent generalisation.
        current_fold = (current_fold + 1) % config.TRAINING.NUM_FOLDS
    dataset["fold_id"] = fold_ids
    dataset = dataset.to_crs(epsg=4326)
    return dataset, dataset["fold_id"].values


def split_train_val_test(X, y, fold_ids, test_fold_id):
    fold_mask = fold_ids == test_fold_id
    X_test, y_test = X[fold_mask], y[fold_mask]
    X_train_val, y_train_val = X[~fold_mask], y[~fold_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=config.TRAINING.VAL_PROP,
        random_state=0,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_class_weights(y_train):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    positive_weight = n_neg / max(n_pos, 1)
    return positive_weight


def cross_validate(dataset):
    precisions, recalls, f1_scores, best_thresholds, num_trees = [], [], [], [], []
    for i in range(config.TRAINING.NUM_FOLDS):
        X, y = prepare_data_for_modelling(dataset)
        dataset, fold_ids = create_spatial_folds(dataset)
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X, y, fold_ids, test_fold_id=i
        )
        pos_weight = calculate_class_weights(y_train)
        model = create_xgboost_model(pos_weight)
        model = train_xgboost_model(model, X_train, y_train, X_val, y_val)
        best_threshold = find_optimal_threshold(model, X_val, y_val)
        y_preds, _ = predict_on_dataset(model, X_test, best_threshold)
        precision, recall, f1 = evaluate_model_performance(y_test, y_preds)
        num_trees.append(model.best_iteration)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        best_thresholds.append(best_threshold)
    return precisions, recalls, f1_scores, best_thresholds, num_trees


def create_spatial_train_test_split(dataset):
    """Create a spatially separated train/test split based on minimum distance."""
    np.random.seed(42)
    dataset = dataset.copy().to_crs(epsg=3857)
    coords = np.vstack([dataset.geometry.x, dataset.geometry.y]).T
    all_indices = np.arange(len(dataset))
    remaining_indices = set(all_indices)
    tree = cKDTree(coords)

    test_indices = []
    num_test_target = int(len(dataset) * config.DATA_SPLIT.TEST_PROP)

    while len(test_indices) < num_test_target and remaining_indices:
        i = np.random.choice(list(remaining_indices))
        test_indices.append(i)
        nearby = tree.query_ball_point(
            coords[i], r=config.DATA_SPLIT.MIN_DISTANCE_BETWEEN_TRAIN_AND_TEST_M
        )
        remaining_indices.difference_update(nearby)

    test_mask = np.zeros(len(dataset), dtype=bool)
    test_mask[test_indices] = True

    test_gdf = dataset[test_mask].to_crs(epsg=4326)
    train_gdf = dataset[~test_mask].to_crs(epsg=4326)
    return train_gdf, test_gdf
