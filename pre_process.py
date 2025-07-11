'''
Pre-processing module for windmill analysis.

* Open datasets
* Reading in and opening turbine coordinates
'''

import xarray as xr
import xroms
import numpy as np

def open_experiment(turb, exp_number, num_of_days):
    '''
    Open datasets as xArray DataSets for a given experiment number.

    Parameters:
    turb (bool): If True, open turbine experiment data. If False, open reference data.
    exp_number (int or str): The experiment number (e.g., 02, 03, 04, etc.).
    num_of_days (int): The number of days (files) to open.

    Returns:
    ds, xgrid: Dataset and grid data from the open files.
    '''
    
    exp_number_str = f'{int(exp_number):02}'
    
    if turb:
        base_path = f'/lustre/storeB/project/nwp/havvind/hav/results/experiment/EXP-{exp_number_str}/norkyst_avg_'
    else:
        base_path = f'/lustre/storeB/project/nwp/havvind/hav/results/reference/REF-{exp_number_str}/norkyst_avg_'
    
    # Construct the file paths
    paths = []
    for i in range(1, num_of_days + 1):
        file_number = f'{i:04}'
        file_path = f'{base_path}{file_number}.nc'
        paths.append(file_path)
    
    # Open the datasets from the constructed paths
    ds = xroms.open_mfnetcdf(paths)
    ds, grid = xroms.roms_dataset(ds, include_cell_volume=True)
    #ds.xroms.set_grid(xgrid)

    return ds, grid

def open_experiment_for_deformation_radius(turb, exp_number, num_of_days, variables):
    '''
    Open datasets for Rossby deformation radius calculation.
    Subsets the datasets to only include relevant variables.

    Parameters:
    turb (bool): If True, open turbine experiment data. If False, open reference data.
    exp_number (int or str): The experiment number (e.g., 02, 03, 04, etc.).
    num_of_days (int): The number of days (files) to open.
    vars (list): List of variable names to include in the dataset.
    '''
    ds = open_experiment(turb, exp_number, num_of_days)

    ds_sub = ds[variables]

    del ds
   
    return ds_sub


def get_turbine_coords(path_to_coords):
    '''
    Function for opening and reading turbine coordinates.
    Returns coordinates as xArray DataSet.

    Input:
    * path_to_coords - 'str'

    Output:
    * turb_coords - 'xr.DataSet'
    '''
    # Reading in the coordinates of turbines

    with open(path_to_coords, 'r') as file:
        lines = file.readlines()[5:]

    # Lists for holding positions of turbines
    lon_turb = []
    lat_turb = []

    # Picking out longitudes and latitudes from the read-in data
    for line in lines:
        lon_lat = line.split()
        if len(lon_lat) == 2:  # Ensure that we have both longitude and latitude
            lon_turb.append(float(lon_lat[0]))
            lat_turb.append(float(lon_lat[1]))
        else:
            raise Exception('Oops, coordinates didnt read in correctly.')

    # Convert to array
    lon_turb = np.array(lon_turb)
    lat_turb = np.array(lat_turb)

    # Convert to xArray DataSet for convenience
    turb_coords = xr.Dataset(
        {
            "coordinates": (("points", "coord"), np.column_stack((lon_turb, lat_turb)))
        },
        coords={
            "points": np.arange(len(lon_turb)),
            "coord": ["Longitude", "Latitude"]
        }
    )

    return turb_coords

def get_windpark_midpoint_indices(ds, target_lon, target_lat):
    '''
    Get the indices of the closest grid point to a target longitude and latitude.
    Parameters:
    ds (xarray.Dataset): The dataset containing the grid.
    lon_target (float): The target longitude.
    lat_target (float): The target latitude.        

    Returns:
    i_loc (int): The index of the closest longitude.
    j_loc (int): The index of the closest latitude.
    '''
    # Find the index of the closest longitude and latitude
    i_loc, j_loc = ds.rho.xroms.argsel2d(target_lon, target_lat)

    return i_loc, j_loc



