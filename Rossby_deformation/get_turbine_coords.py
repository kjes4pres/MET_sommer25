import numpy as np
import xarray as xr

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