import os

from dataset import LoadData
from ConfigHandler import config


def run_dataset_pipeline():
    presence_dataset = LoadData.load_raw_species_data(
        os.path.join(config.DATA.SPECIES.FOLDER, config.DATA.SPECIES.PRESENCE_DATA_PATH)
    )
    background_dataset = LoadData.load_raw_species_data(
        os.path.join(
            config.DATA.SPECIES.FOLDER, config.DATA.SPECIES.BACKGROUND_DATA_PATH
        )
    )
    return presence_dataset, background_dataset
