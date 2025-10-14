import os

import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

from evaluation.Prediction import predict_on_dataset
from preprocessing.EnvironmentData import load_all_environment_variables
from training.Training import prepare_data_for_modelling
from ConfigHandler import config

plt.style.use("seaborn")


def visualise_data_on_map(
    min_lon,
    max_lon,
    min_lat,
    max_lat,
    points_dfs=[],
    points_dfs_colors=[],
    points_dfs_labels=[],
    coastline_alpha=1,
    plot_bounding_box=True,
    limit_to_bounding_box=False,
    accessible_area=None,
    show_plot=True,
):
    coastline = gpd.read_file(
        os.path.join(
            config.DATA.ENVIRONMENTAL.FOLDER,
            config.DATA.ENVIRONMENTAL.COASTLINE_DATA_PATH,
        )
    )
    coastline = coastline.to_crs(epsg=4326)
    coastline.plot(
        color="black",
        linewidth=0.5,
        figsize=(20, 10),
        label="Coastline",
        alpha=coastline_alpha,
    )
    for i, points_df in enumerate(points_dfs):
        plt.scatter(
            points_df["longitude"],
            points_df["latitude"],
            s=5,
            label=points_dfs_labels[i],
            color=points_dfs_colors[i],
        )

    if plot_bounding_box:
        plt.plot(
            [min_lon, min_lon, max_lon, max_lon, min_lon],
            [min_lat, max_lat, max_lat, min_lat, min_lat],
            color="blue",
            alpha=0.5,
            linewidth=1,
            label="Accessible Area",
        )

        # Remove area not well connected to the Med - no observations there anyway
        plt.plot(
            [max_lon, max_lon, max_lon - 6, max_lon - 6, max_lon],
            [min_lat, min_lat + 13.5, min_lat + 13.5, min_lat, min_lat],
            color="green",
            alpha=0.5,
            linewidth=1,
            label="Remove from Accessible Area",
        )

    if accessible_area is not None:
        accessible_area.plot(
            cmap="Blues", alpha=0.4, add_colorbar=False, label="Accessible Area"
        )

    if limit_to_bounding_box:
        plt.xlim(min_lon - 2, max_lon + 2)
        plt.ylim(min_lat - 2, max_lat + 2)

    plt.axis("off")
    if show_plot:
        plt.legend()
        plt.show()


def get_partial_dependence(model, dataset, save_path):
    predictor_cols = config.PREPROCESSING.ENVIRONMENT_DATA
    X, _ = prepare_data_for_modelling(dataset)
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=[i for i in range(len(predictor_cols))],
        feature_names=predictor_cols,
        kind="average",
    )
    plt.savefig(os.path.join(save_path, "partial_dependence.png"))
    plt.close()


def predict_over_accessible_area(model, dataset, accessible_area, save_path):
    accessible_df = accessible_area.to_dataframe().reset_index()

    lat_min = accessible_df["latitude"].min()
    lat_max = accessible_df["latitude"].max()
    lon_min = accessible_df["longitude"].min()
    lon_max = accessible_df["longitude"].max()
    accessible_df = accessible_df[accessible_df["M_accessible"] == 1].reset_index(
        drop=True
    )
    accessible_gdf = gpd.GeoDataFrame(
        accessible_df,
        geometry=gpd.points_from_xy(accessible_df.longitude, accessible_df.latitude),
        crs="EPSG:4326",
    )
    accessible_gdf = load_all_environment_variables(accessible_gdf)
    X, _ = prepare_data_for_modelling(accessible_gdf)
    y_preds, y_probs = predict_on_dataset(model, X, config.PREDICTION.THRESHOLD)
    accessible_gdf["prediction"] = y_preds
    accessible_gdf["prediction_prob"] = y_probs

    accessible_gdf = accessible_gdf.to_crs(epsg=4326)

    visualise_data_on_map(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        [dataset[dataset["label"] == 1], dataset[dataset["label"] == 0]],
        ["green", "red"],
        ["Presence", "Pseudo-absence"],
        plot_bounding_box=True,
        limit_to_bounding_box=True,
        show_plot=False,
    )

    # Convert to raster for colormesh
    lats = np.unique(accessible_area["latitude"].values)
    lons = np.unique(accessible_area["longitude"].values)
    pred_raster = np.full((len(lats), len(lons)), np.nan, dtype=float)
    lat_to_idx = {lat: i for i, lat in enumerate(lats)}
    lon_to_idx = {lon: j for j, lon in enumerate(lons)}
    for _, row in accessible_gdf.iterrows():
        i = lat_to_idx.get(row["latitude"])
        j = lon_to_idx.get(row["longitude"])
        pred_raster[i, j] = row["prediction_prob"]

    plt.gca().pcolormesh(
        lons,
        lats,
        pred_raster,
        cmap="cividis",
        label="Predicted Suitability",
    )
    plt.legend()
    plt.savefig(os.path.join(save_path, "accessible_area_prediction.png"))
    plt.close()
