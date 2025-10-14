import os

import numpy as np
import xarray as xr

from ConfigHandler import config


def obtain_initial_bounding_box(presence_gdf):
    padding = config.PREPROCESSING.ACCESSIBLE_AREA.PADDING
    max_lon = presence_gdf["longitude"].max() + padding
    min_lon = presence_gdf["longitude"].min() - padding
    max_lat = presence_gdf["latitude"].max() + padding
    min_lat = presence_gdf["latitude"].min() - padding
    return min_lon, max_lon, min_lat, max_lat


def filter_background_on_bounding_box(
    background_gdf, min_lon, max_lon, min_lat, max_lat
):
    num_to_remove = (
        (background_gdf["longitude"] < min_lon)
        | (background_gdf["longitude"] > max_lon)
        | (background_gdf["latitude"] < min_lat)
        | (background_gdf["latitude"] > max_lat)
    ).sum()
    background_gdf = background_gdf[
        (background_gdf["longitude"] >= min_lon)
        & (background_gdf["longitude"] <= max_lon)
        & (background_gdf["latitude"] >= min_lat)
        & (background_gdf["latitude"] <= max_lat)
    ].reset_index(drop=True)
    print(
        f"Filtered out {num_to_remove} rows based on bounding box, leaving {background_gdf.shape[0]} rows"
    )
    return background_gdf


def create_accessible_area(min_lon, max_lon, min_lat, max_lat):
    """
    Criteria:
    - Within bounding box of all presence data (and a bit of padding)
    - No land
    - Depth shallower than MAX_DEPTH metres (nudibranchs are generally shallow water species)
    - Remove area only connected to Med via canal (no presence data there anyway)
    TODO: unsure if I can always assume a nan in bathymetry means land, need to check this with Bio-ORACLE docs
    """
    print("Creating accessible area...")
    ds = xr.open_dataset(
        os.path.join(
            config.DATA.ENVIRONMENTAL.FOLDER,
            config.DATA.ENVIRONMENTAL.BIO_ORACLE.FOLDER,
            config.DATA.ENVIRONMENTAL.BIO_ORACLE.BATHYMETRY_FILE,
        )
    )
    bathy = ds["bathymetry_mean"]

    # Limit to bounding box
    bathy = bathy.sel(
        longitude=slice(min_lon, max_lon), latitude=slice(min_lat, max_lat)
    )
    lons = bathy["longitude"].values
    lats = bathy["latitude"].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    # Remove area only connected to Med via canal
    padding = config.PREPROCESSING.ACCESSIBLE_AREA.PADDING
    region_mask = (lon2d < max_lon - padding - 4) | (lat2d > min_lat - padding + 15.5)

    ocean_mask = np.isfinite(bathy.values)[0]
    shelf_mask = (bathy.values >= -config.PREPROCESSING.ACCESSIBLE_AREA.MAX_DEPTH)[0]
    accessible_area_mask = ocean_mask & shelf_mask & region_mask

    accessible_area = xr.DataArray(
        accessible_area_mask.astype(np.uint8),
        coords={"latitude": bathy["latitude"], "longitude": bathy["longitude"]},
        dims=("latitude", "longitude"),
        name="M_accessible",
    )
    print("Accessible area created!")
    return accessible_area


def filter_data_to_be_within_accessible_area(gdf, accessible_area):
    print("Filtering data to be within accessible area...")
    gdf = gdf.to_crs(epsg=4326)

    # Basic fast bounding box filter
    min_lon = accessible_area["longitude"].min().item()
    max_lon = accessible_area["longitude"].max().item()
    min_lat = accessible_area["latitude"].min().item()
    max_lat = accessible_area["latitude"].max().item()
    gdf = gdf[
        (gdf["longitude"] >= min_lon)
        & (gdf["longitude"] <= max_lon)
        & (gdf["latitude"] >= min_lat)
        & (gdf["latitude"] <= max_lat)
    ].reset_index(drop=True)

    # Now filter in more detail based on accessible area mask
    gdf_filtered = gdf[
        gdf.geometry.apply(
            lambda point: accessible_area.sel(
                longitude=point.x, latitude=point.y, method="nearest"
            ).values
            == 1
        )
    ].reset_index(drop=True)
    gdf_removed = gdf[~gdf.index.isin(gdf_filtered.index)].reset_index(drop=True)
    n_removed = gdf.shape[0] - gdf_filtered.shape[0]
    print(
        f"Filtered out {n_removed} rows based on accessible area, leaving {gdf_filtered.shape[0]} rows"
    )
    return gdf_filtered
