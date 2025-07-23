'''
---Kjersti Stangeland, July 2025---

Here the potential density threshold method is used to calculate MLD, with a threshold of 0.03 kgm⁻3. 

'''
from roppy import SGrid
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import griddata
from glob import glob
import time
import xarray as xr
import sys
import os
sys.path.append('/home/kjsta7412/sommer_25/MET_sommer25')
from Rossby_deformation.density import dens

def main():
    """ base_path = glob('/lustre/storeB/project/nwp/havvind/hav/results/reference/')
    months = {
    "02": 27,  # February
    "03": 31,  # March
    "04": 30,  # April
    "05": 31,  # May
    "06": 30   # June
    }

    files=[]

    for month, days in months.items():
        for day in range(1, days + 1): 
            day_str = f"{day:04}"
            file_path = f'/REF-{month}/norkyst_avg_{day_str}.nc'
            files.append(base_path[0]+file_path) """

    files = glob('/lustre/storeB/project/nwp/havvind/hav/results/reference//REF-02/norkyst_avg_0001.nc')

    for f in files:
        print(f)
        calc_mld(f)

def MLD(pot_dens, z):
    """
    Calculate Mixed Layer Depth (MLD) based on potential density profile and depth.

    Parameters:
    - pot_dens: 1D numpy array of potential density [kg/m^3]
    - z: 1D numpy array of corresponding depth levels [m] (negative downward)

    Returns:
    - mld: scalar value of MLD [m], or np.nan if undefined
    """
    # Remove NaNs
    valid = ~np.isnan(pot_dens)
    pot_dens = pot_dens[valid]
    z = z[valid]

    if len(pot_dens) == 0:
        return np.nan

    # Surface density
    surface_density = pot_dens[0]
    threshold = surface_density + 0.03  # MLD is where density exceeds surface + 0.03

    # Find where density exceeds threshold
    exceed = np.where(pot_dens >= threshold)[0]

    if exceed.size == 0:
        return z[-1]  # no depth exceeds threshold

    # Return the first depth where threshold is exceeded
    return z[exceed[0]]

def calc_mld(mfile):
    print(mfile)
    
    ds = Dataset(mfile)
    grid = SGrid(ds) 

    # Depths to interpolate to
    zlevs = np.arange(0,51,1)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(52,102,2), axis =0)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(105,305,5), axis =0)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(520,1020,20), axis =0)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(1050,3050,50), axis =0)

    zlevs = zlevs[np.where(zlevs<=np.max(grid.h))]
    zlevs = np.array(zlevs)*-1.
    
    # Depth of rho-points: z_r
    # Adding zero to end of array (surface)
    z_r = np.insert(grid.z_r, grid.z_r.shape[0], values = np.zeros_like(grid.h), axis=0)
    # Adding local water depth to beginning of array
    z_r = np.insert(z_r, 0, values = -1.*grid.h, axis=0)

    # Variables
    ocean_time = ds.variables['ocean_time']
    salt  = ds.variables['salt'][:]
    temp = ds.variables['temp'][:]
    mask = grid.mask_rho[:,:]

    # Making a file to write to
    #outputf = '/home/kjsta7412/sommer_25/MET_sommer25/output_mld/tests/' + mfile.split('/')[-1].replace('.nc',f'_mld.nc')

    ref_part = mfile.split('reference//')[-1].split('/')[0].replace('-', '_')  # 'REF-02' to 'REF_02'
    norkyst_part = mfile.split('/')[-1][:-3]  # Get 'norkyst_avg_0001' without '.nc'
    filename = f"{ref_part}_{norkyst_part}_mld.nc"
    outputf = '/home/kjsta7412/sommer_25/MET_sommer25/output_mld/tests/' + filename

    rootgrp = Dataset(outputf, 'w')
    time =  rootgrp.createDimension("ocean_time", None)
    X = rootgrp.createDimension("xi_rho", len(ds.dimensions['xi_rho']))
    Y = rootgrp.createDimension("eta_rho", len(ds.dimensions['eta_rho']))
    z = rootgrp.createDimension('z_rho', len(zlevs))

    otime = rootgrp.createVariable("ocean_time","f8",("ocean_time",), zlib=True)
    otime[:] = ocean_time[-1]

    pd = rootgrp.createVariable("pd","f8",("ocean_time", 'z_rho',"eta_rho", "xi_rho",),  zlib=True)
    pd.long_name= "Potential density"
    pd.units = "kg meter-3"

    mld  = rootgrp.createVariable("mld","f8",("ocean_time","eta_rho","xi_rho",),  zlib=True)
    mld.long_name= "Mixed layer depth"
    mld.units = "meter"
    rootgrp.close()

    # Temporary arrays
    tmpd = np.full((1, zlevs.size, salt.shape[2], salt.shape[3]), np.nan)  # Potential density
    tmpm = np.full((1, salt.shape[2], salt.shape[3]), np.nan)  # Mixed layer depth

    ds.close()

    # Looping through grid points
    for y in range(0, salt.shape[2]):
        for x in range(0, salt.shape[3]): 
            if not mask[y, x]:  # skipping land points
               continue

            t = -1
            # Filtering out local water depth on the zlevs
            # Where zlevs is shallower than z_r -> true
            valid_z_mask = zlevs > z_r[:, y, x].min()
            tmpnz = zlevs[valid_z_mask]

            tmpnS = np.insert(salt[t, :, y, x], 0, salt[t, 0, y, x])
            tmpnS = np.append(tmpnS, salt[t, -1, y, x])

            tmpnT = np.insert(temp[t, :, y, x], 0, temp[t, 0, y, x])
            tmpnT = np.append(tmpnT, temp[t, -1, y, x])

            saltZ = griddata(z_r[:, y, x], tmpnS, tmpnz)
            tempZ = griddata(z_r[:, y, x], tmpnT, tmpnz)

            densZ = dens(saltZ, tempZ, np.zeros_like(tempZ))

            dens_profile = np.full(zlevs.shape, np.nan)
            dens_profile[valid_z_mask] = densZ

            tmpd[0, :, y, x] = dens_profile
            tmpm[0, y, x] = MLD(dens_profile, zlevs)
               

    rootgrp = Dataset(outputf, 'r+')
    mld = rootgrp.variables['mld']
    pd = rootgrp.variables['pd']
    mld[:] = tmpm[:]
    pd[:] = tmpd[:]
    rootgrp.close()

    return

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    minutes = int(runtime // 60)
    seconds = runtime % 60
    print(f"Script runtime: {minutes} minutes and {seconds:.4f} seconds")






