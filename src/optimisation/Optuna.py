import optuna

from CrossValidationPipeline import run_cross_validation_pipeline

from ConfigHandler import config


def update_config(params: dict) -> None:
    """Update config params for new optuna experiment"""
    param_config = config.OPTIMISATION.PARAMS
    if param_config.N_ESTIMATORS.USE:
        config.MODEL.N_ESTIMATORS = params["N_ESTIMATORS"]
    if param_config.MAX_DEPTH.USE:
        config.MODEL.MAX_DEPTH = params["MAX_DEPTH"]
    if param_config.LEARNING_RATE.USE:
        config.MODEL.LEARNING_RATE = params["LEARNING_RATE"]
    if param_config.SUBSAMPLE.USE:
        config.MODEL.SUBSAMPLE = params["SUBSAMPLE"]
    if param_config.COLSAMPLE_BYTREE.USE:
        config.MODEL.COLSAMPLE_BYTREE = params["COLSAMPLE_BYTREE"]
    if param_config.REG_LAMBDA.USE:
        config.MODEL.REG_LAMBDA = params["REG_LAMBDA"]
    if param_config.REG_ALPHA.USE:
        config.MODEL.REG_ALPHA = params["REG_ALPHA"]


def objective(trial, dataset):
    """Objective function for optuna optimisation"""
    param_config = config.OPTIMISATION.PARAMS
    params = {}
    if param_config.N_ESTIMATORS.USE:
        params["N_ESTIMATORS"] = trial.suggest_int(
            "N_ESTIMATORS",
            param_config.N_ESTIMATORS.MIN,
            param_config.N_ESTIMATORS.MAX,
            step=param_config.N_ESTIMATORS.STEP,
        )
    if param_config.MAX_DEPTH.USE:
        params["MAX_DEPTH"] = trial.suggest_int(
            "MAX_DEPTH",
            param_config.MAX_DEPTH.MIN,
            param_config.MAX_DEPTH.MAX,
            step=param_config.MAX_DEPTH.STEP,
        )
    if param_config.LEARNING_RATE.USE:
        params["LEARNING_RATE"] = trial.suggest_float(
            "LEARNING_RATE",
            param_config.LEARNING_RATE.MIN,
            param_config.LEARNING_RATE.MAX,
            step=param_config.LEARNING_RATE.STEP,
        )
    if param_config.SUBSAMPLE.USE:
        params["SUBSAMPLE"] = trial.suggest_float(
            "SUBSAMPLE",
            param_config.SUBSAMPLE.MIN,
            param_config.SUBSAMPLE.MAX,
            step=param_config.SUBSAMPLE.STEP,
        )
    if param_config.COLSAMPLE_BYTREE.USE:
        params["COLSAMPLE_BYTREE"] = trial.suggest_float(
            "COLSAMPLE_BYTREE",
            param_config.COLSAMPLE_BYTREE.MIN,
            param_config.COLSAMPLE_BYTREE.MAX,
            step=param_config.COLSAMPLE_BYTREE.STEP,
        )
    if param_config.REG_LAMBDA.USE:
        params["REG_LAMBDA"] = trial.suggest_float(
            "REG_LAMBDA",
            param_config.REG_LAMBDA.MIN,
            param_config.REG_LAMBDA.MAX,
            step=param_config.REG_LAMBDA.STEP,
        )
    if param_config.REG_ALPHA.USE:
        params["REG_ALPHA"] = trial.suggest_float(
            "REG_ALPHA",
            param_config.REG_ALPHA.MIN,
            param_config.REG_ALPHA.MAX,
            step=param_config.REG_ALPHA.STEP,
        )

    update_config(params)

    metrics = run_cross_validation_pipeline(dataset)
    print(metrics)

    objective = -1.0 * metrics["cv"]["mean_f1"]
    if objective != objective:
        objective = 0
    return objective


def optimise_hyperparameters_via_optuna(dataset):
    if config.OPTIMISATION.USE_RANDOM_SAMPLER:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    else:
        study = optuna.create_study()
    if not config.OPTIMISATION.USE_OPTUNA:
        return study

    study.enqueue_trial(
        {
            "N_ESTIMATORS": 2000,
            "MAX_DEPTH": 5,
            "LEARNING_RATE": 0.03,
            "SUBSAMPLE": 0.8,
            "COLSAMPLE_BYTREE": 0.8,
            "REG_LAMBDA": 1.0,
            "REG_ALPHA": 0.0,
        }
    )

    study.optimize(
        lambda trial: objective(trial, dataset),
        n_trials=config.OPTIMISATION.N_TRIALS,
    )
    update_config(study.best_trial.params)
    return study
