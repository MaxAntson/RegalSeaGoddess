import pandas as pd

from ConfigHandler import config


def load_raw_species_data(data_path):
    df = pd.read_csv(data_path, sep="\t", on_bad_lines="skip")
    print(f"Species Counts: {df['species'].nunique()}")
    print(f"Initial number of rows: {df.shape[0]}")
    return df
