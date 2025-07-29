import numpy as np
from glob import glob
import os
import sys
sys.path.append('/home/kjsta7412/sommer_25/MET_sommer25')
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import cartopy.feature as cfeature
from roppy import SGrid
from netCDF4 import Dataset
from Rossby_deformation.get_turbine_coords import get_turbine_coords
from Rossby_deformation.funcs import *

mplstyle.use(['ggplot', 'fast'])

path = '/lustre/storeB/project/nwp/havvind/hav/results/experiment/EXP-03/norkyst_avg_0001.nc'
fid = Dataset(path)
grid = SGrid(fid)
del fid

# Opening the turbine coordinates as a xArray DataSet
sorvest_F = get_turbine_coords('/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Sorvest_F.txt')
nordvest_C = get_turbine_coords('/lustre/storeB/project/nwp/havvind/turbine_coordinates/windfarms_Nordvest_C.txt')

# Square around wind park
min_lon_SV = np.min(sorvest_F.coordinates[:,0].values)
min_lat_SV = np.min(sorvest_F.coordinates[:,1].values)
max_lon_SV = np.max(sorvest_F.coordinates[:,0].values)
max_lat_SV = np.max(sorvest_F.coordinates[:,1].values)

area_lon_SV = [min_lon_SV, max_lon_SV]
area_lat_SV = [min_lat_SV, max_lat_SV]

# Square around wind park
min_lon_NV = np.min(nordvest_C.coordinates[:,0].values)
min_lat_NV = np.min(nordvest_C.coordinates[:,1].values)
max_lon_NV = np.max(nordvest_C.coordinates[:,0].values)
max_lat_NV = np.max(nordvest_C.coordinates[:,1].values)

area_lon_NV = [min_lon_NV, max_lon_NV]
area_lat_NV = [min_lat_NV, max_lat_NV]

# Extracting paths to files containing the computed Rossby deformation radius
# Note: I'm using the reference datasets and not the experiments
# Rossby deformation radius computed from the experiment datasets are found in output_bdr/EXP

filefolder = glob('/lustre/storeB/project/nwp/havvind/hav/analysis_kjsta/output_bdr/REF')

# Only using June because we want the largest Rossby radius
months = {
"06": 30   # June
}

files=[]  # empty list to store paths in

# building paths to contain each daily file and named thereafter
for month, days in months.items():
    for day in range(1, days + 1): 
        day_str = f"{day:04}"
        file_path = f'/REF_{month}_norkyst_avg_{day_str}_brr.nc'
        files.append(filefolder[0]+file_path)

# Internal Rossby radius of June - from area of wind parks
R1_june_SV = monthly_mean_area(files, grid, area_lon_SV, area_lat_SV)
R1_june_SV = R1_june_SV.gamma_r

# Internal Rossby radius of June - from area of wind parks
R1_june_NV = monthly_mean_area(files, grid, area_lon_NV, area_lat_NV)
R1_june_NV = R1_june_NV.gamma_r

zlevs = np.arange(0,51,1)
zlevs = np.insert(zlevs,len(zlevs),values=np.arange(52,102,2), axis =0)
zlevs = np.insert(zlevs,len(zlevs),values=np.arange(105,305,5), axis =0)
zlevs = np.insert(zlevs,len(zlevs),values=np.arange(520,1020,20), axis =0)
zlevs = np.insert(zlevs,len(zlevs),values=np.arange(1050,3050,50), axis =0)

zlevs = zlevs[np.where(zlevs<=np.max(grid.h))]
zlevs = np.array(zlevs)*-1.


# Extracting filepaths to MLD for the reference runs

filefolder = glob('/lustre/storeB/project/nwp/havvind/hav/analysis_kjsta/output_mld/REF')

months = {
"02": 27,  # February
"03": 31,  # March
"04": 30,  # April
"05": 31,  # May
"06": 30   # June
}

files_ref=[]  # empty list to store paths in

# building paths to contain each daily file and named thereafter
for month, days in months.items():
    for day in range(1, days + 1): 
        day_str = f"{day:04}"
        file_path = f'/REF_{month}_norkyst_avg_{day_str}_mld.nc'
        files_ref.append(filefolder[0]+file_path)


# Extracting filepaths to MLD for the experiments

filefolder = glob('/lustre/storeB/project/nwp/havvind/hav/analysis_kjsta/output_mld/EXP')

months = {
"02": 27,  # February
"03": 31,  # March
"04": 30,  # April
"05": 31,  # May
"06": 30   # June
}

files_exp=[]  # empty list to store paths in

# building paths to contain each daily file and named thereafter
for month, days in months.items():
    for day in range(1, days + 1): 
        day_str = f"{day:04}"
        file_path = f'/EXP_{month}_norkyst_avg_{day_str}_mld.nc'
        files_exp.append(filefolder[0]+file_path)

# MLD and potential dens at Sørvest-F - study area is 60R1x60R1 around wind farm.
# Area lon/lat is the extent of the study area
lon_SV, lat_SV, ref_SV = make_study_area(files_ref, grid, area_lon_SV, area_lat_SV, R1_june_SV)
lon_SV, lat_SV, exp_SV = make_study_area(files_exp, grid, area_lon_SV, area_lat_SV, R1_june_SV)

# MLD and potential dens at Sørvest-F - study area is 60R1x60R1 around wind farm.
# Area lon/lat is the extent of the study area
lon_NV, lat_NV, ref_NV = make_study_area(files_ref, grid, area_lon_NV, area_lat_NV, R1_june_NV)
lon_NV, lat_NV, exp_NV = make_study_area(files_exp, grid, area_lon_NV, area_lat_NV, R1_june_NV)

# Monthly mean of MLD
mmean_mld_ref_SV = ref_SV.mld.resample(ocean_time='1M').mean(dim='ocean_time')
mmean_mld_exp_SV = exp_SV.mld.resample(ocean_time='1M').mean(dim='ocean_time')

# Monthly mean of MLD
mmean_mld_ref_NV = ref_NV.mld.resample(ocean_time='1M').mean(dim='ocean_time')
mmean_mld_exp_NV = exp_NV.mld.resample(ocean_time='1M').mean(dim='ocean_time')

months = ['Feb', 'Mar', 'Apr', 'May', 'Jun']

# Horizontal mean of MLD
hmmean_SV_mld_ref = mmean_mld_ref_SV.mean(dim=['eta_rho', 'xi_rho'])
hmmean_SV_mld_exp = mmean_mld_exp_SV.mean(dim=['eta_rho', 'xi_rho'])

# Horizontal mean of MLD
hmmean_NV_mld_ref = mmean_mld_ref_NV.mean(dim=['eta_rho', 'xi_rho'])
hmmean_NV_mld_exp = mmean_mld_exp_NV.mean(dim=['eta_rho', 'xi_rho'])

# Monthly mean of potential density of the area
mmean_pd_ref_SV = ref_SV.pd.resample(ocean_time='1M').mean(dim='ocean_time')
mmean_pd_exp_SV = exp_SV.pd.resample(ocean_time='1M').mean(dim='ocean_time')

# Horizontal mean
hmmean_pd_SV_ref = mmean_pd_ref_SV.mean(dim=['eta_rho', 'xi_rho'])
hmmean_pd_SV_exp = mmean_pd_exp_SV.mean(dim=['eta_rho', 'xi_rho'])

# Monthly mean of potential density of the area
mmean_pd_ref_NV = ref_NV.pd.resample(ocean_time='1M').mean(dim='ocean_time')
mmean_pd_exp_NV = exp_NV.pd.resample(ocean_time='1M').mean(dim='ocean_time')

# Horizontal mean
hmmean_pd_NV_ref = mmean_pd_ref_NV.mean(dim=['eta_rho', 'xi_rho'])
hmmean_pd_NV_exp = mmean_pd_exp_NV.mean(dim=['eta_rho', 'xi_rho'])

print('it works so far')
print('starting the plotting')

fig, ax = plt.subplots(2, 5, figsize=(20, 8))

n_xticks = 3

for i, axs in enumerate(ax[0]):
    axs.plot(hmmean_pd_SV_ref[i, :].values, zlevs, color='teal', label='REF')
    axs.plot(hmmean_pd_SV_exp[i, :].values, zlevs, color='darkgoldenrod', label='EXP')
    axs.set_title(months[i])
    axs.legend()
    axs.axhline(y=hmmean_SV_mld_ref[i], color='teal', alpha=0.6, linestyle='--', linewidth=2)
    axs.axhline(y=hmmean_SV_mld_exp[i], color='darkgoldenrod', alpha=0.6, linestyle='--', linewidth=2)
    axs.grid()
    #axs.set_xticks(np.linspace(axs.get_xlim()[0], axs.get_xlim()[1], n_xticks))

print('Done with first row of subplots')
print('starting second row')

for i, axs in enumerate(ax[1]):
    axs.plot(hmmean_pd_NV_ref[i, :].values, zlevs, color='teal', label='REF')
    axs.plot(hmmean_pd_NV_exp[i, :].values, zlevs, color='darkgoldenrod', label='EXP')
    axs.set_title(months[i])
    axs.legend()
    axs.axhline(y=hmmean_NV_mld_ref[i], color='teal', alpha=0.6, linestyle='--', linewidth=2)
    axs.axhline(y=hmmean_NV_mld_exp[i], color='darkgoldenrod', alpha=0.6, linestyle='--', linewidth=2)
    axs.grid()
    #axs.set_xticks(np.linspace(axs.get_xlim()[0], axs.get_xlim()[1], n_xticks))

print('done with second row')
print('adding titles and saving figure')

# Add row titles
fig.text(0.5, 0.94, 'Sørvest-F', ha='center', va='center', fontsize=14, fontweight='bold')
fig.text(0.5, 0.48, 'Nordvest-C', ha='center', va='center', fontsize=14, fontweight='bold')

fig.suptitle('Potential density - Sørvest-F and Nordvest-C', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('/home/kjsta7412/sommer_25/MET_sommer25/Figures/pd_mld_SVF_NVC.png')