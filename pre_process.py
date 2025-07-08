'''
Pre-processing module for windmill analysis.

* Open datasets
* Reading in and opening turbine coordinates
* Computing the Rossby radius
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
    ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)

    return ds, xgrid


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


def rossby_radius(f, rho, z_w, grid):
    '''
    Calculate the first Rossby radius using the WKB approximation.
    
    Parameters:
    f (float or ndarray): The Coriolis parameter
    rho (ndarray): The density profile as a function of depth, which 
                   can vary in space.
    z_w (ndarray): The depth levels
    grid (object): The grid object 

    Returns:
    R (ndarray): The calculated Rossby radius at the specified depth levels. 
                 The output is an array corresponding to the spatial 
                 dimensions defined in the grid.
    '''
    # Calculate buoyancy frequency squared
    N2 = xroms.N2(rho, grid)
    N2 = N2.fillna(0)
    
    # Take the square root to get buoyancy frequency
    N = np.sqrt(N2)

    # Interpolating from s-layer to z-depths (in meters)
    depths = np.linspace(-100, 0) 
    N_sliced = N.xroms.zslice(grid, depths, z=z_w)
    N_sliced = N_sliced.fillna(0)

    # Buoyancy frequency integrated over depth
    N_int = N_sliced.integrate(coord='z_w')
    
    # WKB approx. of Rossby radius
    R = (1 / (np.abs(f) * np.pi)) * N_int

    R = R.mean(dim='ocean_time')

    return R