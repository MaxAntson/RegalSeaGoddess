from ConfigHandler import config

from CrossValidationPipeline import run_cross_validation_pipeline
from DatasetPipeline import run_dataset_pipeline
from DataSplitPipeline import run_data_split_pipeline
from EvaluationPipeline import run_evaluation_pipeline
from ExperimentHandler import ExperimentHandler
from InterpretationPipeline import run_interpretation_pipeline
from OptimisationPipeline import run_optimisation_pipeline
from PreprocessPipeline import run_preprocessing_pipeline
from ProductionPipeline import run_production_pipeline
from TrainingPipeline import run_training_pipeline
from ResultsHandler import ResultsHandler


def add_config_params_for_current_experiment(experiment_title, experiment_description):
    config.EXPERIMENT_TITLE = experiment_title
    config.EXPERIMENT_DESCRIPTION = experiment_description


def run_experiment(experiment_title: str = "", experiment_description: str = ""):
    print(f"Running experiment: {experiment_title}")
    add_config_params_for_current_experiment(experiment_title, experiment_description)

    experiment_handler = ExperimentHandler(experiment_title=experiment_title)

    outputs_save_path = experiment_handler.results_path
    experiment_handler.save_config(config)

    # Get presence/absence data
    presence_data, background_data = run_dataset_pipeline()

    # Filtering, adding environmental variables, creating accessible area, bias correction
    dataset, accessible_area = run_preprocessing_pipeline(
        presence_data, background_data
    )

    # Train/test for final evaluation - splitting with spatial blocking
    train_dataset, test_dataset = run_data_split_pipeline(dataset, outputs_save_path)

    # Optuna optimisation of hyperparameters - e.g. max_depth - maximising F1 score in cross validation
    run_optimisation_pipeline(dataset=train_dataset)
    experiment_handler.save_config(config)

    # Post-optimisation, run a final cross-validation training + evaluation
    print("Running final cross validation")
    cv_metrics = run_cross_validation_pipeline(dataset=train_dataset)
    rh = ResultsHandler(results_path=outputs_save_path)
    rh.add_multiple_metrics(metrics=cv_metrics)

    # Train and test the final model based on optimised hyperparams
    pred_model = run_training_pipeline(dataset=train_dataset)
    experiment_handler.save_config(config)
    metrics = run_evaluation_pipeline(dataset=test_dataset, model=pred_model)
    rh.add_multiple_metrics(metrics=metrics)
    rh.save_results()

    # Save some plots (e.g. accessible area prediction plot, partial dependency plot)
    # and useful interpretations like feature importance
    run_interpretation_pipeline(
        model=pred_model,
        dataset=test_dataset,
        accessible_area=accessible_area,
        save_path=outputs_save_path,
    )

    # Production-ready files to plug into inference code
    run_production_pipeline(model=pred_model, save_path=outputs_save_path)
    experiment_handler.save_config(config)
