import os

import geopandas as gpd
import xarray as xr

from ConfigHandler import config


def get_variable_based_on_lat_and_lon(ds, var_name, lat, lon):
    return (
        ds[var_name]
        .sel(time="1970-01-01", longitude=lon, latitude=lat, method="nearest")
        .values.item()
    )


def load_distance_to_shore(gdf):
    if "distance_to_shore_m" not in config.PREPROCESSING.ENVIRONMENT_DATA:
        return gdf
    gdf = gdf.to_crs(epsg=3857)
    coastline = gpd.read_file(
        os.path.join(
            config.DATA.ENVIRONMENTAL.FOLDER,
            config.DATA.ENVIRONMENTAL.COASTLINE_DATA_PATH,
        )
    )
    coastline = coastline.to_crs(epsg=3857)
    gdf["distance_to_shore_m"] = gdf.geometry.apply(
        lambda point: coastline.distance(point).min()
    )
    gdf = gdf.to_crs(epsg=4326)
    return gdf


def load_all_environment_variables(gdf):
    print("Loading in environment variables...")
    variables = config.PREPROCESSING.ENVIRONMENT_DATA
    for var in variables:
        if var == "distance_to_shore_m":
            continue
        path = os.path.join(
            config.DATA.ENVIRONMENTAL.FOLDER,
            config.DATA.ENVIRONMENTAL.BIO_ORACLE.FOLDER,
            f"{var}.nc",
        )
        ds = xr.open_dataset(path)
        var_name = list(ds.data_vars.keys())[0]

        # TODO: maybe this is faster but needs some correction
        # gds = gpd.GeoDataFrame(
        #     ds.to_dataframe().reset_index(),
        #     geometry=gpd.points_from_xy(ds.longitude, ds.latitude),
        #     crs="EPSG:4326",
        # )
        # gdf = gpd.sjoin_nearest(gdf, ds, how="left")
        # gdf.drop(
        #     columns=["index_right", "lon", "lat"],
        #     inplace=True,
        # )

        gdf[var] = gdf.apply(
            lambda row: get_variable_based_on_lat_and_lon(
                ds, var_name, row["latitude"], row["longitude"]
            ),
            axis=1,
        )
        ds.close()
    gdf = load_distance_to_shore(gdf)

    if config.PREPROCESSING.DROP_NA_ENVIRONMENTAL:
        gdf = gdf.dropna(subset=variables).reset_index(drop=True)
    print("Finished loading in environment variables!")
    return gdf
