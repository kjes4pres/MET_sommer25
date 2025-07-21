import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xroms
from datetime import datetime


def find_indices_of_point(grid, target_lon, target_lat):
    '''
    Finds the [y, x] indices of grid point closest to a given coordinate point (lon, lat).
    The model grid is lon(y, x), lat(y, x).
    '''
    lon = grid.lon_rho
    lat = grid.lat_rho

    y, x = xroms.argsel2d(lon, lat, target_lon, target_lat)

    return y, x


def R1_rolling_mean_all(files):
    '''
    Opens all daily files of Rossby deformation radius.
    Returns a DataArray of the rolling mean with time window 7 days.
    '''
    # Opening all files
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    ocean_time = full_ds.ocean_time.values
    datetime_array = [datetime.fromtimestamp(ot) for ot in ocean_time]
    full_ds['ocean_time'] = datetime_array

    # Computing the rolling mean of rossby radius
    R1_roll = full_ds.gamma_r.rolling(ocean_time=7).mean()

    return R1_roll


def extract_R1_all(files):
    '''
    Opens all daily files of Rossby deformation radius.
    Returns DataArray of size (149, 1148, 2747)
    '''
    # Opening all files
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    ocean_time = full_ds.ocean_time.values
    datetime_array = [datetime.fromtimestamp(ot) for ot in ocean_time]
    full_ds['ocean_time'] = datetime_array

    # Computing the rolling mean of rossby radius
    R1 = full_ds.gamma_r

    return R1


def monthly_mean(files):
    '''
    Opens all daily files of Rossby deformation radius.
    Resamples and take the monthly mean. Time dim going from 149->5

    '''
    # Opening all files
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    ocean_time = full_ds.ocean_time.values
    datetime_array = [datetime.fromtimestamp(ot) for ot in ocean_time]
    full_ds['ocean_time'] = datetime_array

    ds_monthly = full_ds.resample(ocean_time='1M').mean(dim='ocean_time')

    
    return ds_monthly


def monthly_mean_area(files, grid, area_lon, area_lat):
    '''
    Opens all daily files of Rossby deformation radius.
    Returns the horizontal mean of the radius of a given area.
    Area chosen around the windpark.
    '''
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    ocean_time = full_ds.ocean_time.values
    datetime_array = [datetime.fromtimestamp(ot) for ot in ocean_time]
    full_ds['ocean_time'] = datetime_array

    j1, i1 = find_indices_of_point(grid, area_lon[0], area_lat[0])
    j2, i2 = find_indices_of_point(grid, area_lon[1], area_lat[0])
    j3, i3 = find_indices_of_point(grid, area_lon[1], area_lat[1])
    j4, i4 = find_indices_of_point(grid, area_lon[0], area_lat[1])

    j_min = np.min([j1, j2, j3, j4])
    j_max = np.max([j1, j2, j3, j4])

    i_min = np.min([i1, i2, i3, i4])
    i_max = np.max([i1, i2, i3, i4])
    
    ds_area = full_ds.isel(eta_rho=slice(j_min, j_max), xi_rho=slice(i_min, i_max))
    R1_area_mean = ds_area.mean(dim=['eta_rho', 'xi_rho'])

    return R1_area_mean.gamma_r


def move_distance_from_point(start_lon, start_lat, distance):
    '''
    Move a given distance in meters
    north, west, south, east from a coordinate point.
    '''
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


def make_region_ds(files, grid, area_lon, area_lat):
    '''
    Makes a geographic subset of datasets.
    area_lon:[min_lon, max_lon]
    area_lat: [min_lat, max_lat]
    '''
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    ocean_time = full_ds.ocean_time.values
    datetime_array = [datetime.fromtimestamp(ot) for ot in ocean_time]
    full_ds['ocean_time'] = datetime_array

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

def make_study_area(files, grid, area_lon, area_lat, R1):
    '''
    Makes a subset of dataset.
    Takes in a box around the windpark and adds 5*R1 in all directions.
    '''
    full_ds = xr.open_mfdataset(files)

    # Converting ocean_time from seconds since initialization to datetime values
    ocean_time = full_ds.ocean_time.values
    datetime_array = [datetime.fromtimestamp(ot) for ot in ocean_time]
    full_ds['ocean_time'] = datetime_array

    # original box around windpark
    point_1 = (area_lon[1], area_lat[1])
    point_2 = (area_lon[0], area_lat[1])
    point_3 = (area_lon[0], area_lat[0])
    point_4 = (area_lon[1], area_lat[0])

    # distance in meters to expand box with
    distance = R1*5

    # east, west, north, south coordinates if moving distance in every direction from point
    e1, w1, n1, s1 = move_distance_from_point(point_1[0], point_1[1], distance)
    e2, w2, n2, s2 = move_distance_from_point(point_2[0], point_2[1], distance)
    e3, w3, n3, s3 = move_distance_from_point(point_3[0], point_3[1], distance)
    e4, w4, n4, s4 = move_distance_from_point(point_4[0], point_4[1], distance)

    # New points - expanded box
    point_1_exp = (e1, n1)
    point_2_exp = (w2, n2)
    point_3_exp = (w3, s3)
    point_4_exp = (e4, s4)


    j1, i1 = find_indices_of_point(grid, point_1_exp[0], point_1_exp[1])
    j2, i2 = find_indices_of_point(grid, point_2_exp[0], point_2_exp[1])
    j3, i3 = find_indices_of_point(grid, point_3_exp[0], point_3_exp[1])
    j4, i4 = find_indices_of_point(grid, point_4_exp[0], point_4_exp[1])

    j_min = np.min([j1, j2, j3, j4])
    j_max = np.max([j1, j2, j3, j4])

    i_min = np.min([i1, i2, i3, i4])
    i_max = np.max([i1, i2, i3, i4])
    
    ds_area = full_ds.isel(eta_rho=slice(j_min, j_max), xi_rho=slice(i_min, i_max))

    return ds_area

