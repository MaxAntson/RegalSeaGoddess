from optimisation.Optuna import optimise_hyperparameters_via_optuna


def run_optimisation_pipeline(dataset):
    optimise_hyperparameters_via_optuna(dataset)
