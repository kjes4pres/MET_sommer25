'''
For plotting:)
'''

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean.cm as cmo
import numpy as np
import xarray as xr
import xroms
import cartopy 



def plot_map_Rossby_radius(ds, R, windpark, exp_name, time_idx, i_loc, j_loc):
    '''
    Function to plot the Rossby radius of a given time index on a map.

    Parameters:
    ds (xarray.Dataset): The dataset containing the grid information.
    R (xarray.DataArray): The Rossby radius data to be plotted.
    windpark (xarray.Dataset): The dataset containing the wind park coordinates.
    exp_name (str): The name of the experiment for the plot title.
    time_idx (int or None): The specific time index to plot. 
    '''
    if time_idx is not None:
        R = R.isel(ocean_time=time_idx)  # Select the specific time index


    proj = ccrs.NorthPolarStereo()
    fig, ax = plt.subplots(1,2, figsize = (10, 6), subplot_kw={'projection':proj}, constrained_layout=True)
    plt.subplots_adjust(wspace=0.05)
    cmap='cmo.deep'  

    mask = ds.mask_rho.values
    masked_data = np.ma.masked_array(R.values, mask=(mask == 0))  # Masking land values
    R_masked = xr.DataArray(masked_data, coords=R.coords, dims=R.dims)

    ax[0].set_extent([4, 6, 56, 57.5], crs=ccrs.PlateCarree())

    plott = R_masked.plot(ax = ax[0], x='lon_rho', y='lat_rho', transform=ccrs.PlateCarree(), cmap=cmap, 
                          vmin=0, vmax=3000, add_colorbar=False)  # zoomed in plot
    cbar = plt.colorbar(plott, ax=ax[0], orientation='vertical', pad=0.05, aspect=50)
    cbar.set_label('Rossby deformation radius (m)', labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    plot = R_masked.plot(ax = ax[1], x='lon_rho', y='lat_rho', transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False)
    cbar = plt.colorbar(plot, ax=ax[1], orientation='vertical', pad=0.05, aspect=50)
    cbar.set_label('Rossby deformation radius (m)', labelpad=20)
    cbar.ax.tick_params(labelsize=10)


    for axs in ax.flatten():
        for i in range(len(windpark.coordinates)):
            axs.plot(windpark.coordinates[i][0], windpark.coordinates[i][1], transform = ccrs.PlateCarree(), color = 'black', marker ='*', markersize=3) 
    
        gl = axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='lightgray', alpha=0.5, linestyle='--')
        gl.top_labels = False  # Disable top labels
        gl.right_labels = False  # Disable right labels
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

    ax[0].plot(ds.lon_rho[i_loc, j_loc], ds.lat_rho[i_loc, j_loc], transform=ccrs.PlateCarree(), color='white', marker='o', markersize=5, label='Midpoint Sørvest F')

    fig.suptitle(f'Rossby deformation radius {exp_name}\n Sørvest F location')
    plt.show()