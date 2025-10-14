import warnings

from Pipeline import run_experiment
from ConfigHandler import config

warnings.filterwarnings("ignore")

run_experiment(
    experiment_title="first_run",
    experiment_description="Ensuring the pipeline can run.",
)
