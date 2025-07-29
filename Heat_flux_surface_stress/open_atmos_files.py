import xarray as xr
import os

months = {
"02": 28,  # February (not a leap year in 2022)
"03": 31,  # March
"04": 30,  # April
"05": 31,  # May
"06": 30   # June
         }

times = ["00", "03", "09", "12", "15", "18", "21"]

# Making a list of paths to turbine datasets
base_path_turb = '/lustre/storeB/project/nwp/havvind/atmosphere/turbines/2022'
files_turb = []

for month, days in months.items():
    for day in range(1, days + 1):
        day_str = f"{day:02}"  # Formatting day as two digits
        for time in times:
            # Construct the file name
            file_name_turb = f'fc2022{month}{day_str}{time}_turbines.nc'
            # Construct the full file path
            file_path = os.path.join(base_path_turb, month, day_str, time, file_name_turb)
            files_turb.append(file_path)

# Making list of reference datasets
base_path_ref = '/lustre/storeB/project/nwp/havvind/atmosphere/reference/2022'
files_ref = []

for month, days in months.items():
    for day in range(1, days + 1):
        day_str = f"{day:02}"  # Formatting day as two digits
        for time in times:
            # Construct the file name
            file_name_ref= f'fc2022{month}{day_str}{time}_reference.nc'
            # Construct the full file path
            file_path = os.path.join(base_path_ref, month, day_str, time, file_name_ref)
            files_ref.append(file_path)
    

def open_dataset_heatflux_surfstress(files):
    '''
    Opens a list of atmospheric output files.
    Returns xArray DataSet with wanted variables
    for investigating heat fluxes and surface stress.
    '''
    full_ds = xr.open_mfdataset(files)

    variables = ['SFX_H', 'SFX_LE', 'SFX_FMV',
                 'integral_of_surface_net_downward_shortwave_flux_wrt_time',
                 'integral_of_surface_net_downward_longwave_flux_wrt_time',
                 'integral_of_surface_net_downward_latent_heat_evaporation_flux_wrt_time',
                 'integral_of_surface_downward_sensible_heat_flux_wrt_time']
    
    ds_sub = full_ds[variables]

    return ds_sub
    




