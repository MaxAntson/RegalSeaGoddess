import xgboost as xgb

from ConfigHandler import config


def create_xgboost_model(positive_weight):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=config.MODEL.N_ESTIMATORS,
        max_depth=config.MODEL.MAX_DEPTH,
        learning_rate=config.MODEL.LEARNING_RATE,
        subsample=config.MODEL.SUBSAMPLE,
        colsample_bytree=config.MODEL.COLSAMPLE_BYTREE,
        reg_lambda=config.MODEL.REG_LAMBDA,
        reg_alpha=config.MODEL.REG_ALPHA,
        scale_pos_weight=positive_weight,
        n_jobs=4,
        early_stopping_rounds=config.MODEL.EARLY_STOPPING_ROUNDS,
    )
    return model


def train_xgboost_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model
