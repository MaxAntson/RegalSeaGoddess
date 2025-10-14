import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from ConfigHandler import config


def filter_dataset_1_by_distance_to_dataset_2(gdf_1, gdf_2, threshold_m):
    gdf_1 = gdf_1.to_crs(epsg=3857)
    gdf_2 = gdf_2.to_crs(epsg=3857)
    joined = gdf_1.sjoin_nearest(gdf_2[["geometry"]], how="left", distance_col="dist_m")
    joined = joined.drop_duplicates("gbifID")
    gdf_1 = gdf_1.loc[joined["dist_m"] >= threshold_m].copy()
    gdf_1 = gdf_1.to_crs(epsg=4326).reset_index(drop=True)
    gdf_2 = gdf_2.to_crs(epsg=4326).reset_index(drop=True)
    return gdf_1


def create_raw_raster(gdf, accessible_area):
    raster = np.zeros_like(accessible_area, dtype=int)
    lons = accessible_area["longitude"].values
    lats = accessible_area["latitude"].values
    print("Creating raw raster for background sampling...")
    for lon, lat in tqdm(zip(gdf["longitude"], gdf["latitude"])):
        lon_idx = np.abs(lons - lon).argmin()
        lat_idx = np.abs(lats - lat).argmin()
        if (
            accessible_area.sel(longitude=lon, latitude=lat, method="nearest").values
            == 0
        ):
            continue
        raster[lat_idx, lon_idx] += 1
    # Expect large number in a hotspot, median of 0 as most points are empty, but still a decently large amount of non-zero points
    print(
        f"Raster stats - max: {raster.max()}, median: {np.median(raster)}, non-zero count: {(raster > 0).sum()}"
    )
    return raster


def create_probability_raster(raw_raster, sigma=1):
    prob_raster = gaussian_filter(raw_raster.astype(float), sigma=sigma)
    prob_raster /= prob_raster.sum()
    return prob_raster


def sample_background_points(accessible_area, background_gdf):
    bg_raster = create_raw_raster(background_gdf, accessible_area)
    bg_prob_raster = create_probability_raster(bg_raster, sigma=1)

    lons = accessible_area["longitude"].values
    lats = accessible_area["latitude"].values

    np.random.seed(42)
    background_points = np.random.choice(
        np.arange(bg_prob_raster.size),
        size=config.PREPROCESSING.BG_SAMPLE_SIZE,
        replace=True,
        p=bg_prob_raster.flatten(),
    )
    background_indices = np.unravel_index(background_points, bg_prob_raster.shape)
    background_lons = lons[background_indices[1]]
    background_lats = lats[background_indices[0]]
    bg_vars = pd.DataFrame({"latitude": background_lats, "longitude": background_lons})
    gbg_vars = gpd.GeoDataFrame(
        bg_vars,
        geometry=gpd.points_from_xy(bg_vars.longitude, bg_vars.latitude),
        crs="EPSG:4326",
    )
    return gbg_vars
