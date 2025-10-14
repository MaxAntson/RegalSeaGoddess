from ConfigHandler import config


def get_feature_importance(model):
    importances = model.get_booster().get_score(importance_type="gain")
    importance_df = {}
    for i, var in enumerate(config.PREPROCESSING.ENVIRONMENT_DATA):
        importance_df[var] = importances[f"f{i}"]
    return importance_df
