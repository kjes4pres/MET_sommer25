'''
A lil dumpster for code that was a part of the testing
process but is not used anymore.
This file is not used in the current workflow, but it may
be useful for future reference or debugging.

Just some code snippets I am too emotionally attached to delete.
'''


import xarray as xr
import xroms
import dask.array as da
from scipy.interpolate import griddata
import numpy as np
from scipy.integrate import simps
from numba import njit, prange


# Plot of domain and windpark location
'''
proj = ccrs.Mercator()
fig, ax = plt.subplots(1,2, figsize = (10, 6), subplot_kw={'projection':proj}, constrained_layout=True)

ax[0].set_extent([4, 6, 56, 57.5], crs=ccrs.PlateCarree())

test.z_rho.isel(ocean_time=0, s_rho=-1).plot(ax = ax[0], x='lon_rho', y='lat_rho', transform=ccrs.PlateCarree(), add_colorbar=False)
test.z_rho.isel(ocean_time=0, s_rho=-1).plot(ax = ax[1], x='lon_rho', y='lat_rho', transform=ccrs.PlateCarree())

for axs in ax.flatten():
    for i in range(len(sorvest_F.coordinates)):
        axs.plot(sorvest_F.coordinates[i][0], sorvest_F.coordinates[i][1], transform = ccrs.PlateCarree(), color = 'black', marker ='*', markersize=3) 

    gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='lightgray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # Disable top labels
    gl.right_labels = False  # Disable right labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

fig.suptitle('Sørvest F location')
'''

# Different approaches to calculate the Rossby radius

def interpolate_density_on_depths(dens, z_r, depths, mask, time_index):
    """
    Interpolate density from s-layers to specified z-levels for one time step.

    Parameters:
    dens: numpy.ndarray
        Density array with shape (time, vertical levels, y, x)
    z_r: numpy.ndarray
        Depth array with shape (time, vertical levels, y, x)
    depths: numpy.ndarray
        Target depth levels for interpolation
    mask: numpy.ndarray
        Boolean mask for valid grid points
    time_index: int
        The index of the time step to process

    Returns:
    numpy.ndarray
        Interpolated density at specified z-levels with NaNs where depths are not valid
    """
    # Initialize a temporary array for the interpolated densities
    tmp_density = np.full((len(depths), dens.shape[2], dens.shape[3]), np.nan)
    
    for y in range(0, dens.shape[2]):
        for x in range(0, dens.shape[3]):
            # Skip invalid grid points
            if not mask[y, x]:
                continue
            
            # Get depth points and densities for the specified time index
            depth_points = z_r[time_index, :, y, x]
            density_values = dens[time_index, :, y, x]
            z_min = depth_points.min()  # Minimum depth for the current grid cell
            
            # Check for valid depths that are greater than z_min
            valid_depths = depths[depths > z_min]

            # Ensure we have enough valid levels to interpolate
            if valid_depths.size == 0:
                print(f"No valid depths for grid point ({y}, {x}) with z_min = {z_min}.")
                continue
            
            # Perform interpolation
            try:
                density_interpolated = griddata(depth_points, density_values, valid_depths)

                # Fill the appropriate positions in tmp_density with the interpolated values
                # Iterate over the valid_depths and assign corresponding values
                for i, z_level in enumerate(valid_depths):
                    if z_level in depths:  # Only fill if z_level exists in depths
                        # Get index in depths
                        idx = np.where(depths == z_level)[0][0]
                        tmp_density[idx, y, x] = density_interpolated[i]  # Fill the respective depth
            except Exception as e:
                print(f"Error during interpolation for grid point ({y}, {x}): {e}")

    return tmp_density  # Return your interpolated densities

 
def dens_to_depths(dens, z_r, depths, mask):
    """
    Interpolate density from s-layers to specified z-levels for all time steps.

    Parameters:
    dens: numpy.ndarray
        Density array with shape (time, vertical levels, y, x)
    z_r: numpy.ndarray
        Depth array with shape (time, vertical levels, y, x)
    depths: numpy.ndarray
        Target depth levels for interpolation
    mask: numpy.ndarray
        Boolean mask for valid grid points

    Returns:
    numpy.ndarray
        Interpolated density at specified z-levels for all time steps
    """
    time_steps = dens.shape[0]
    result_density = np.full((time_steps, depths.shape[0], dens.shape[2], dens.shape[3]), np.nan)

    for t in range(time_steps):
        print(f"Interpolating densities for time step {t}...")
        result_density[t] = interpolate_density_on_depths(dens, z_r, depths, mask, t)

    return result_density  # Return all interpolated densities for all time steps



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
    R = np.zeros(dens.shape[2:])  

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

