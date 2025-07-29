from datetime import datetime, timedelta
from glob import glob

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyresample
import xarray as xr
import xroms
from netCDF4 import Dataset
from roppy import SGrid
from scipy.interpolate import griddata


def find_indices_of_point(grid, target_lon, target_lat):
    """
    Finds the [y, x] indices of grid point closest to a given coordinate point (lon, lat).
    The model grid is lon(y, x), lat(y, x).
    """
    lon = grid.lon_rho
    lat = grid.lat_rho

    y, x = xroms.argsel2d(lon, lat, target_lon, target_lat)

    return y, x


def convert_ocean_time(full_ds):
    """
    Converts ocean_time from seconds since initialization
    to datetime values and assigns it back to the dataset.
    """
    ocean_time = full_ds.ocean_time.values
    datetime_array = [datetime.fromtimestamp(ot) for ot in ocean_time]
    full_ds["ocean_time"] = datetime_array
    return full_ds


def R1_rolling_mean_all(files):
    """
    Opens all daily files of Rossby deformation radius.
    Returns a DataArray of the rolling mean with time window 7 days.
    """
    # Opening all files
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    full_ds = convert_ocean_time(full_ds)

    # Computing the rolling mean of rossby radius
    R1_roll = full_ds.gamma_r.rolling(ocean_time=7).mean()

    return R1_roll


def extract_R1_all(files):
    """
    Opens all daily files of Rossby deformation radius.
    Returns DataArray of size (149, 1148, 2747)
    """
    # Opening all files
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    full_ds = convert_ocean_time(full_ds)

    # Computing the rolling mean of rossby radius
    R1 = full_ds.gamma_r

    return R1


def monthly_mean(files):
    """
    Opens all daily files of Rossby deformation radius.
    Resamples and take the monthly mean. Time dim going from 149->5

    """
    # Opening all files
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    full_ds = convert_ocean_time(full_ds)

    ds_monthly = full_ds.resample(ocean_time="1M").mean(dim="ocean_time")

    return ds_monthly


def monthly_mean_area(files, grid, area_lon, area_lat):
    """
    Opens all daily files of Rossby deformation radius.
    Returns the horizontal mean of the radius of a given area.
    Area chosen around the windpark.
    """
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    full_ds = convert_ocean_time(full_ds)

    # Getting the indices of the area
    j1, i1 = find_indices_of_point(grid, area_lon[0], area_lat[0])
    j2, i2 = find_indices_of_point(grid, area_lon[1], area_lat[0])
    j3, i3 = find_indices_of_point(grid, area_lon[1], area_lat[1])
    j4, i4 = find_indices_of_point(grid, area_lon[0], area_lat[1])

    j_min = np.min([j1, j2, j3, j4])
    j_max = np.max([j1, j2, j3, j4])

    i_min = np.min([i1, i2, i3, i4])
    i_max = np.max([i1, i2, i3, i4])

    # Subsetting
    ds_area = full_ds.isel(eta_rho=slice(j_min, j_max), xi_rho=slice(i_min, i_max))
    R1_area_mean = ds_area.mean(dim=["eta_rho", "xi_rho"])
    R1_area_mon_mean = R1_area_mean.resample(ocean_time="1M").mean(dim="ocean_time")

    return R1_area_mon_mean


def move_distance_from_point(start_lon, start_lat, distance):
    """
    Move a given distance in meters
    north, west, south, east from a coordinate point.
    """
    lat_rad = np.radians(start_lat)
    meters_per_degree_longitude = 111321.4 * np.cos(lat_rad)
    delta_lon = distance / meters_per_degree_longitude
    delta_lat = distance / 111320

    # Move east
    east_lon = start_lon + delta_lon

    # Move west
    west_lon = start_lon - delta_lon

    # Move north
    north_lat = start_lat + delta_lat

    # Move south
    south_lat = start_lat - delta_lat

    return east_lon, west_lon, north_lat, south_lat


# ----- These 3 functions go together-----
# Masking method


def make_region_ds(files, grid, area_lon, area_lat):
    """
    Subsets dataset to a given geographic area using lat/lon masking.
    """
    full_ds = xr.open_mfdataset(files)

    # Convert ocean_time to datetime
    full_ds = convert_ocean_time(full_ds)

    # Load grid lat/lon
    # fid = Dataset('/lustre/storeB/project/nwp/havvind/hav/results/reference/REF-02/norkyst_avg_0001.nc')
    # grid = SGrid(fid)

    lon = np.asarray(grid.lon_rho)
    lat = np.asarray(grid.lat_rho)
    area_lon = np.asarray(area_lon)
    area_lat = np.asarray(area_lat)

    full_ds = full_ds.assign_coords(
        lon_rho=(["eta_rho", "xi_rho"], lon), lat_rho=(["eta_rho", "xi_rho"], lat)
    )

    mask = (
        (lon >= area_lon[0])
        & (lon <= area_lon[1])
        & (lat >= area_lat[0])
        & (lat <= area_lat[1])
    )

    mask_da = xr.DataArray(mask, dims=("eta_rho", "xi_rho"))

    # Apply mask
    masked_da = full_ds.where(mask_da, drop=True)

    return area_lon, area_lat, masked_da


def expand_box_around_farm(area_lon, area_lat, R1):
    """
    Expand bounding box (lon/lat) by R1 meters in all directions.
    Returns: new_area_lon, new_area_lat
    """
    R1 = R1.values

    center_lat = (area_lat[0] + area_lat[1]) / 2
    lat_rad = np.radians(center_lat)
    meters_per_degree_longitude = 111321.4 * np.cos(lat_rad)

    delta_lon = (R1 * 25) / meters_per_degree_longitude
    delta_lat = (R1 * 25) / 111320

    new_area_lon = [area_lon[0] - delta_lon, area_lon[1] + delta_lon]
    new_area_lat = [area_lat[0] - delta_lat, area_lat[1] + delta_lat]

    return new_area_lon, new_area_lat


def make_study_area(files, grid, area_lon, area_lat, R1):
    new_area_lon, new_area_lat = expand_box_around_farm(area_lon, area_lat, R1)
    return make_region_ds(files, grid, new_area_lon, new_area_lat)


# ------------------------------------------------------
# Below function works with expand_area_around_farm
# Slicing method
def rotate_point(grid, point, angle):
    """
    Rotate a point (lon, lat) around the origin (0, 0) based on the specified angle.
    """
    y, x = find_indices_of_point(grid, point[0], point[1])

    angle = angle[y, x]
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    return rotation_matrix.dot(point)


def make_region_dataset(files, grid, area_lon, area_lat, R1):
    """
    Creates a geographic subset of datasets.

    If R1 is provided, expands the area by 5*R1 around the given box (for wind park).
    Otherwise, uses the given lon/lat bounds directly.

    area_lon: [min_lon, max_lon]
    area_lat: [min_lat, max_lat]
    """

    full_ds = xr.open_mfdataset(files)
    full_ds = convert_ocean_time(full_ds)

    fid = Dataset(
        "/lustre/storeB/project/nwp/havvind/hav/results/reference/REF-02/norkyst_avg_0001.nc"
    )
    grid = SGrid(fid)

    lon = grid.lon_rho
    lat = grid.lat_rho

    full_ds = full_ds.assign_coords(
        lon_rho=(["eta_rho", "xi_rho"], lon), lat_rho=(["eta_rho", "xi_rho"], lat)
    )

    # Expand area if R1 is given
    if R1 is not None:
        area_lon, area_lat = expand_box_around_farm(area_lon, area_lat, R1)

    # Find grid indices of points
    j1, i1 = find_indices_of_point(grid, area_lon[0], area_lat[0])
    j2, i2 = find_indices_of_point(grid, area_lon[1], area_lat[0])
    j3, i3 = find_indices_of_point(grid, area_lon[1], area_lat[1])
    j4, i4 = find_indices_of_point(grid, area_lon[0], area_lat[1])

    j_min = np.min([j1, j2, j3, j4])
    j_max = np.max([j1, j2, j3, j4])
    i_min = np.min([i1, i2, i3, i4])
    i_max = np.max([i1, i2, i3, i4])

    ds_area = full_ds.isel(eta_rho=slice(j_min, j_max), xi_rho=slice(i_min, i_max))

    return ds_area


# Victors approach -----
def v2_make_region_dataset(files, grid, area_lon, area_lat, R1):
    """
    Creates a geographic subset of datasets.

    If R1 is provided, expands the area by 5*R1 around the given box (for wind park).
    Otherwise, uses the given lon/lat bounds directly.

    area_lon: [min_lon, max_lon]
    area_lat: [min_lat, max_lat]
    """

    full_ds = xr.open_mfdataset(files)
    full_ds = convert_ocean_time(full_ds)

    fid = Dataset(
        "/lustre/storeB/project/nwp/havvind/hav/results/reference/REF-02/norkyst_avg_0001.nc"
    )
    grid = SGrid(fid)

    lon = grid.lon_rho
    lat = grid.lat_rho

    full_ds = full_ds.assign_coords(
        lon_rho=(["eta_rho", "xi_rho"], lon), lat_rho=(["eta_rho", "xi_rho"], lat)
    )

    # Expand area
    area_lon_exp, area_lat_exp = expand_box_around_farm(area_lon, area_lat, R1)

    ds_geo = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
    pos_geo = pyresample.geometry.SwathDefinition(
        lons=[area_lon[0], area_lon[1], area_lon[1], area_lon[0]],
        lats=[area_lat[0], area_lat[0], area_lat[1], area_lat[1]],
    )

    (
        _,
        valid_output_index,
        index_array,
        distance_array,
    ) = pyresample.kd_tree.get_neighbour_info(
        source_geo_def=ds_geo,
        target_geo_def=pos_geo,
        radius_of_influence=800,
        neighbours=1,
    )

    index_array_2d = np.unravel_index(index_array, ds_geo.shape)
    (eta_indices, xi_indices) = index_array_2d[0], index_array_2d[1]

    eta_min, eta_max = eta_indices.min(), eta_indices.max()  # 256–603
    xi_min, xi_max = xi_indices.min(), xi_indices.max()  # 19–368

    cropped_ds = full_ds.isel(
        eta_rho=slice(eta_min, eta_max + 1), xi_rho=slice(xi_min, xi_max + 1)
    )

    dlat = 0.005  # ~1 km at mid-latitudes
    dlon = 0.005

    lat = np.arange(area_lat_exp[0], area_lat_exp[1] + dlat, dlat)
    lon = np.arange(area_lon_exp[0], area_lon_exp[1] + dlon, dlon)

    # Meshgrid of expanded area
    lon2d, lat2d = np.meshgrid(lon, lat)

    mod_grd = np.stack(
        [cropped_ds.lon_rho.values.ravel(), cropped_ds.lat_rho.values.ravel()], -1
    )
    interp_dir = griddata(
        mod_grd, cropped_ds.gamma_r.values.ravel(), (lon2d, lat2d), method="nearest"
    )

    return lon2d, lat2d, interp_dir
