# Datasets Used

Due to the regal sea goddess being a rather slow animal, only fixed variables and 10 year averages were used for all environmental data for this initial version. In the future, it would be useful to build this out with daily/monthly data instead.

## Presence / Absence Data

- All data downloaded from [GBIF](https://www.gbif.org/)
- `raw_rsg_data.csv`
  - Presence data of the regal sea goddess
  - Citation: GBIF.org (4 September 2025) GBIF Occurrence Download <https://doi.org/10.15468/dl.a7bqyb>
- `background_data.csv`
  - This is the background dataset to create a probability map for generating background data, to tackle locational sampling bias and have higher likelihood of generating a more reliable absence dataset rather than selecting random data points.
  - It contains the species subclasses `Echinodermata`, `Cnidaria`, and `Gastropoda`, the following:
    - It helps tackle sampling bias by showcasing where people usually go to spot similar species
    - It helps to define more likely true absence points -> places where people go often and spot similar animals, but never ever spot a regal sea goddess, are more likely to be absence rather than random data points
  - Citation: GBIF.org (5 September 2025) GBIF Occurrence Download <https://doi.org/10.15468/dl.ey3c22>

## Environmental Data

All data in `environmental/` in the `data` folder.

- `ne_10m_coastline/`
  - stored coastline data to within 10m resolution, downloaded from [NaturalEarthData](https://www.naturalearthdata.com/downloads/10m-physical-vectors/)
  - used for map visualisation and (crudely) calculating distance to shore
- `bio_oracle/`
  - For all datasets, `Present Day Conditions (2010-2020)` and `Surface Layers` were selected on the [Bio-ORACLE](https://www.bio-oracle.org/downloads-to-email.php) website
  - The following variables were taken:
    - Mean Bathymetry (Depth) - `bio_oracle/bathymetry.nc`
    - Mean Topographic Slope  - `bio_oracle/slope.nc`
    - Mean Sea Surface Temperature  - `bio_oracle/mean_sst.nc`
    - Range Sea Surface Temperature  - `bio_oracle/range_sst.nc`
    - Mean Chlorophyll-a  - `bio_oracle/chlorophyll.nc`
    - Mean Salinity  - `bio_oracle/salinity.nc`
