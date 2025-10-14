import os

from training.Training import create_spatial_train_test_split

from ConfigHandler import config


def run_data_split_pipeline(dataset, save_path):
    train_dataset, test_dataset = create_spatial_train_test_split(dataset)
    stats = [
        "--- Training dataset ---",
        f"Size: {len(train_dataset)}",
        f"Presence points: {(train_dataset['label'] == 1).sum()}",
        f"Background points: {(train_dataset['label'] == 0).sum()}",
        "\n--- Testing dataset ---",
        f"Size: {len(test_dataset)}",
        f"Presence points: {(test_dataset['label'] == 1).sum()}",
        f"Background points: {(test_dataset['label'] == 0).sum()}",
    ]
    stats_text = "\n".join(stats)
    with open(os.path.join(save_path, "data_split_stats.txt"), "w") as f:
        f.write(stats_text)
    return train_dataset, test_dataset
