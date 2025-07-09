'''
Pre-processing module for windmill analysis.

* Open datasets
* Reading in and opening turbine coordinates
* Computing the Rossby radius
'''

import xarray as xr
import xroms
import dask.array as da
from scipy.interpolate import griddata
import numpy as np
from scipy.integrate import simps
from numba import njit, prange


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
    i_loc, j_loc = ds.temp.xroms.argsel2d(target_lon, target_lat)

    return i_loc, j_loc



def interpolate_density_on_zlevs(dens, z_r, zlevs, mask, time_index):
    """
    Interpolate density from s-layers to specified z-levels for one time step.

    Parameters:
    dens: numpy.ndarray
        Density array with shape (time, vertical levels, y, x)
    z_r: numpy.ndarray
        Depth array with shape (time, vertical levels, y, x)
    zlevs: numpy.ndarray
        Target depth levels for interpolation
    mask: numpy.ndarray
        Boolean mask for valid grid points
    time_index: int
        The index of the time step to process

    Returns:
    numpy.ndarray
        Interpolated density at specified z-levels with NaNs where zlevs are not valid
    """
    # Initialize a temporary array for the interpolated densities
    tmp_density = np.full((len(zlevs), dens.shape[2], dens.shape[3]), np.nan)
    
    for y in range(0, dens.shape[2]):
        for x in range(0, dens.shape[3]):
            # Skip invalid grid points
            if not mask[y, x]:
                continue
            
            # Get depth points and densities for the specified time index
            depth_points = z_r[time_index, :, y, x]
            density_values = dens[time_index, :, y, x]
            z_min = depth_points.min()  # Minimum depth for the current grid cell
            
            # Check for valid zlevs that are greater than z_min
            valid_zlevs = zlevs[zlevs > z_min]

            # Ensure we have enough valid levels to interpolate
            if valid_zlevs.size == 0:
                print(f"No valid zlevs for grid point ({y}, {x}) with z_min = {z_min}.")
                continue
            
            # Perform interpolation
            try:
                density_interpolated = griddata(depth_points, density_values, valid_zlevs)

                # Fill the appropriate positions in tmp_density with the interpolated values
                # Iterate over the valid_zlevs and assign corresponding values
                for i, z_level in enumerate(valid_zlevs):
                    if z_level in zlevs:  # Only fill if z_level exists in zlevs
                        # Get index in zlevs
                        idx = np.where(zlevs == z_level)[0][0]
                        tmp_density[idx, y, x] = density_interpolated[i]  # Fill the respective depth
            except Exception as e:
                print(f"Error during interpolation for grid point ({y}, {x}): {e}")

    return tmp_density  # Return your interpolated densities

 
def dens_to_zlevs(dens, z_r, zlevs, mask):
    """
    Interpolate density from s-layers to specified z-levels for all time steps.

    Parameters:
    dens: numpy.ndarray
        Density array with shape (time, vertical levels, y, x)
    z_r: numpy.ndarray
        Depth array with shape (time, vertical levels, y, x)
    zlevs: numpy.ndarray
        Target depth levels for interpolation
    mask: numpy.ndarray
        Boolean mask for valid grid points

    Returns:
    numpy.ndarray
        Interpolated density at specified z-levels for all time steps
    """
    time_steps = dens.shape[0]
    result_density = np.full((time_steps, zlevs.shape[0], dens.shape[2], dens.shape[3]), np.nan)

    for t in range(time_steps):
        print(f"Interpolating densities for time step {t}...")
        result_density[t] = interpolate_density_on_zlevs(dens, z_r, zlevs, mask, t)

    return result_density  # Return all interpolated densities for all time steps



@njit
def N2(rho_, z_, rho_0=1000.0):
    '''
    --!Function stolen from Ann Kristin Sperrevik!--
    Return the stratification frequency

    Parameters
    ----------
    rho : array_like
        density [kg/m^3]
    z : array_like
        depths [m] (positive upward)

    Returns
    -------
    N2 : array_like
        Stratification frequency [1/s], where the vertical dimension (the
        first dimension) is one less than the input arrays.
    '''
    rho = np.asarray(rho_)
    z = np.asarray(z_)

    assert rho.shape == z.shape, 'rho and z must have the same shape.'
    r_z = np.diff(rho, axis=0) / np.diff(z, axis=0)

    buoy_freq_sq =  -(9.8 / rho_0) * r_z

    return buoy_freq_sq

@njit
def calc_rossby_radius(dens, z, f):
    """
    Calculate the Rossby radius based on buoyancy frequency N integrated over depth.

    Parameters
    ----------
    dens : array_like
        Density profile [kg/m^3] with shape (cz, y, x), cz represents depth levels.
    z : array_like
        Depth levels corresponding to density [m] (should be positive upward).
    f : array_like
        Coriolis parameter [1/s] with the same dimensions as y and x.

    Returns
    -------
    R : array_like
        Rossby radius [m] with the same dimensions as f.
    """
    # Initialize an array for Rossby radius results
    R = np.zeros(dens.shape[2:])  # Shape same as mask or original grid

    # Calculate N^2 for all depth levels
    n2 = np.full((dens.shape[0], dens.shape[1]-1, dens.shape[2], dens.shape[3]), np.nan)  # Initialize with NaNs
    
    for y in range(dens.shape[2]):  # Iterate over y
        for x in range(dens.shape[3]):  # Iterate over x
            for i in range(dens.shape[0]):  # Iterate over time
                # Calculate N^2 only for valid density values
                if not np.isnan(dens[i, :, y, x]).all():
                 n2[i, :, y, x] = N2(dens[i, :, y, x], z)

    # Avoid negative values
    n2[n2 < 0] = 0.0000001

    # Calculate buoyancy frequency N
    N = np.sqrt(n2)

    # Integration over depth using Simpson's rule for each grid point
    for y in range(N.shape[2]):
        for x in range(N.shape[3]):
            for i in range(N.shape[0]):
            # Check for valid N values
                if not np.isnan(N[i, :, y, x]).all():
                    integral = simps(N[i, :, y, x], z)  # Integrating N with respect to z
                
                # Calculate Rossby radius
                if np.abs(f[y, x]) > 0:
                    R[y, x] = integral / (np.abs(f[y, x]) * np.pi)
                else:
                    R[y, x] = np.nan  # Handle case when f is zero or very small

    return R  # Return the computed Rossby radius


def get_rossby_radius(rho, z, f):
    # Buoyancy frequency squared
    buoy_freq_sq = N2(rho, z, rho_0=1000.0)
    
    # Check for and handle NaN values in buoy_freq_sq
    buoy_freq_sq = np.nan_to_num(buoy_freq_sq)  # Replace NaN with 0

    # Take the square root to get buoyancy frequency
    N = np.sqrt(buoy_freq_sq)

    # Now we need to integrate over the depths correctly
    # Midpoints of the depth intervals for integration
    z_mid = (z[:-1] + z[1:]) / 2  # Midpoints for the integration

    # Ensure N has the shape that corresponds to z_mid
    N = N[:-1]  # Trimming N to match z_mid shape for integration

    # Buoyancy frequency integrated over depth using midpoints
    N_int = np.trapz(N, z_mid, axis=0)  # Integrate over the depth dimension
    
    f = np.asarray(f)

    # WKB approx. of Rossby radius
    R = (1 / (np.abs(f) * np.pi)) * N_int

    return R


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
    depths = np.linspace(-1000, 0) 
    N_sliced = N.xroms.zslice(grid, depths, z=z_w)
    N_sliced = N_sliced.fillna(0)

    # Buoyancy frequency integrated over depth
    N_int = N_sliced.integrate(coord='z_w')
    
    # WKB approx. of Rossby radius
    R = (1 / (np.abs(f) * np.pi)) * N_int

    #R = R.mean(dim='ocean_time')

    return R
