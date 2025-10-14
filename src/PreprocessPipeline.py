from preprocessing import AccessibleArea, BackgroundSampling, EnvironmentData, General


def run_preprocessing_pipeline(presence_df, background_df):
    # Basic filtering for both sets
    presence_df = General.filter_basic_issues_from_dataset(presence_df)
    background_df = General.filter_basic_issues_from_dataset(background_df)
    presence_gdf = General.convert_to_geodataframe(presence_df)
    background_gdf = General.convert_to_geodataframe(background_df)

    # Accessible area
    min_lon, max_lon, min_lat, max_lat = AccessibleArea.obtain_initial_bounding_box(
        presence_gdf
    )
    background_gdf = AccessibleArea.filter_background_on_bounding_box(
        background_gdf, min_lon, max_lon, min_lat, max_lat
    )
    accessible_area = AccessibleArea.create_accessible_area(
        min_lon, max_lon, min_lat, max_lat
    )
    presence_gdf = AccessibleArea.filter_data_to_be_within_accessible_area(
        presence_gdf, accessible_area
    )
    background_gdf = AccessibleArea.filter_data_to_be_within_accessible_area(
        background_gdf, accessible_area
    )
    background_gdf = General.filter_dataset_1_by_distance_to_dataset_2(
        background_gdf, presence_gdf
    )

    # Load environment data - doing this before picking points to remove NaNs earlier
    presence_gdf = EnvironmentData.load_all_environment_variables(presence_gdf)

    # Bias correction incl pseudo-absence selection
    presence_gdf = General.spatially_thin(presence_gdf)
    background_gdf = BackgroundSampling.sample_background_points(
        accessible_area, background_gdf
    )
    background_gdf = EnvironmentData.load_all_environment_variables(background_gdf)

    gdf = General.combine_presence_and_background_into_single_gdf(
        presence_gdf, background_gdf
    )
    return gdf, accessible_area
