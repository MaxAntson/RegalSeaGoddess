import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ConfigHandler import config


def filter_basic_issues_from_dataset(df):
    num_to_remove = (
        df["coordinateUncertaintyInMeters"]
        >= config.PREPROCESSING.COORD_UNCERTAINTY_THRESHOLD
    ).sum()
    df = df[
        (
            df["coordinateUncertaintyInMeters"]
            < config.PREPROCESSING.COORD_UNCERTAINTY_THRESHOLD
        )
        | (df["coordinateUncertaintyInMeters"].isna())
    ].reset_index(drop=True)
    print(
        f"Filtered out {num_to_remove} rows based on coordinate uncertainty, leaving {df.shape[0]} rows"
    )

    # Remove nans for co-ord uncertainty
    if config.PREPROCESSING.DROP_NA_COORD_UNCERTAINTY:
        num_to_remove = df["coordinateUncertaintyInMeters"].isna().sum()
        df = df[~df["coordinateUncertaintyInMeters"].isna()].reset_index(drop=True)
        print(
            f"Filtered out {num_to_remove} rows based on missing coordinate uncertainty, leaving {df.shape[0]} rows"
        )

    # Remove rows with no lat/lon
    num_to_remove = (df["decimalLatitude"].isna() | df["decimalLongitude"].isna()).sum()
    df = df[
        ~(df["decimalLatitude"].isna()) & ~(df["decimalLongitude"].isna())
    ].reset_index(drop=True)
    print(
        f"Filtered out {num_to_remove} rows based on missing lat/lon, leaving {df.shape[0]} rows"
    )
    df = df.rename(
        columns={"decimalLatitude": "latitude", "decimalLongitude": "longitude"}
    )
    return df


def convert_to_geodataframe(df):
    df = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    return df


def spatially_thin(gdf):
    """
    Spatially thin a dataset so that no points are within MIN_DISTANCE_M of each other.
    Reduces sampling bias for the 'presence' data.

    TODO: Improve distance calculations to account for spherical nature of the Earth.
    """
    gdf = gdf.to_crs(epsg=3857)
    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T

    tree = cKDTree(coords)
    to_keep = np.ones(len(gdf), dtype=bool)
    for i in range(len(coords)):
        if to_keep[i]:
            neighbors = tree.query_ball_point(
                coords[i], r=config.PREPROCESSING.SPATIAL_THINNING_MIN_DISTANCE_M
            )
            neighbors.remove(i)
            to_keep[neighbors] = False
    print(
        f"Spatially thinned dataset from {len(gdf)} to {to_keep.sum()} points to reduce sampling bias"
    )
    gdf = gdf.to_crs(epsg=4326)
    return gdf[to_keep].reset_index(drop=True)


def combine_presence_and_background_into_single_gdf(presence_gdf, background_gdf):
    presence_gdf["label"] = 1
    background_gdf["label"] = 0
    presence_gdf = presence_gdf.to_crs(epsg=4326)
    background_gdf = background_gdf.to_crs(epsg=4326)

    combined_gdf = gpd.GeoDataFrame(
        pd.concat([presence_gdf, background_gdf], ignore_index=True),
        crs=presence_gdf.crs,
    )
    combined_gdf = combined_gdf.sample(
        frac=1, replace=False, random_state=42
    ).reset_index(drop=True)
    print(
        f"Combined presence and background data into single GeoDataFrame with {combined_gdf.shape[0]} rows"
    )
    return combined_gdf


def filter_dataset_1_by_distance_to_dataset_2(gdf_1, gdf_2):
    """TODO: Improve distance calculations to account for spherical nature of the Earth."""
    gdf_1 = gdf_1.to_crs(epsg=3857)
    gdf_2 = gdf_2.to_crs(epsg=3857)
    joined = gdf_1.sjoin_nearest(gdf_2[["geometry"]], how="left", distance_col="dist_m")
    joined = joined.drop_duplicates("gbifID")
    gdf_1 = gdf_1.loc[
        joined["dist_m"]
        >= config.PREPROCESSING.MIN_DISTANCE_BETWEEN_PRESENCE_AND_ABSENCE_M
    ].copy()
    gdf_2 = gdf_2.to_crs(epsg=4326)
    gdf_1 = gdf_1.to_crs(epsg=4326).reset_index(drop=True)
    return gdf_1
