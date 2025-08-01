import xarray as xr
import numpy as np
from roppy import SGrid
from netCDF4 import Dataset
from glob import glob
import time

def main():
    """ base_path = glob('/lustre/storeB/project/nwp/havvind/hav/results/analysis_kjsta/output_mld/REF/')
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
            file_path = f'/REF_{month}/norkyst_avg_{day_str}_mld.nc'
            files.append(base_path[0]+file_path) """

    files = glob('/lustre/storeB/project/nwp/havvind/hav/analysis_kjsta/output_mld/REF/REF_02_norkyst_avg_0001_mld.nc')
    zeta_files = glob('/lustre/storeB/project/nwp/havvind/hav/results/reference/REF-02/norkyst_avg_0001.nc')

    for f in files:
        for z in zeta_files:
            print(f, z)
            calc_pea(f, z)


def calc_pea(mfile, zfile):
    '''
    ...
    '''

    print(mfile)
    
    ds = Dataset(mfile)

    ds2 = Dataset(zfile)
    grid = SGrid(ds2) 

    # Depths to interpolate to
    zlevs = np.arange(0,51,1)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(52,102,2), axis =0)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(105,305,5), axis =0)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(520,1020,20), axis =0)
    zlevs = np.insert(zlevs,len(zlevs),values=np.arange(1050,3050,50), axis =0)

    zlevs = zlevs[np.where(zlevs<=np.max(grid.h))]
    zlevs = np.array(zlevs)*-1.

    # Variables
    ocean_time = ds.variables['ocean_time']
    rho  = ds.variables['pd'][:] + 1000
    mask = grid.mask_rho[:,:]
    zeta = ds2.variables['zeta'][:]
    g = 9.81
    rhoref = 1027.0
    maxdepth = ds.variables['mld'][:]

    # Depth of rho-points: z_r
    # Adding zero to end of array (surface)
    z_r = np.insert(grid.z_r, grid.z_r.shape[0], values = np.zeros_like(grid.h), axis=0)
    # Adding free surface height to z_r
    z_r[-1] = zeta[0, :, :]
    # Adding local water depth to beginning of array
    z_r = np.insert(z_r, 0, values = -1.*grid.h, axis=0)


    # Making a file to write to
    base_filename = mfile.split('/')[-1][:-3]

    if base_filename.endswith('_mld'):
        base_filename = base_filename[:-4]

    filename = f"{base_filename}_pea_try.nc"
    outputf = '/lustre/storeB/project/nwp/havvind/hav/analysis_kjsta/output_pea/tests/' + filename

    rootgrp = Dataset(outputf, 'w')
    time =  rootgrp.createDimension("ocean_time", None)
    X = rootgrp.createDimension("xi_rho", len(ds.dimensions['xi_rho']))
    Y = rootgrp.createDimension("eta_rho", len(ds.dimensions['eta_rho']))

    otime = rootgrp.createVariable("ocean_time","f8",("ocean_time",), zlib=True)
    otime[:] = ocean_time[-1]

    pea = rootgrp.createVariable("pea","f8",("ocean_time","eta_rho", "xi_rho",),  zlib=True)
    pea.long_name= "Potential energy anomaly"
    pea.units = "meter3 seconds-2"

    rootgrp.close()

    # Temporary arrays
    tmpea = np.full((1, rho.shape[2], rho.shape[3]), np.nan)  # Potential energy anom.

    ds.close()

    # Looping through grid points
    for y in range(0, rho.shape[2]):
        for x in range(0, rho.shape[3]): 
            if not mask[y, x]:  # skipping land points
               continue

            tmpea[0, y, x] = 0
            tmpmaxdepth = maxdepth[0, y, x]

            # Filtering out local water depth on the zlevs
            # Where zlevs is shallower than z_r -> true
            valid_z_mask = zlevs > maxdepth[0, y, x]
            tmpz = zlevs[valid_z_mask]

            # Matching rho with valid z levels
            tmprho = rho[0, :tmpz.shape[0], y, x]

            z_depth = (tmpz[0:-1] + tmpz[1:]) / 2

            # Integrand
            P = g*((rhoref-tmprho)/rhoref)*tmpz

            # Total depth 
            H = zeta[0, y, x] - tmpz.min()

            # Integrating rho over z-levels
            for k in range(0, len(z_depth) - 1):
                tmpea[0, y, x] = tmpea[0, y, x] + np.abs(z_depth[k] - z_depth[k + 1]) / 2.0 * (P[k] + P[k + 1])
                tmpea[0, y, x] = tmpea[0, y, x]/H


    rootgrp = Dataset(outputf, 'r+')
    pea = rootgrp.variables['pea']
    pea[:] = tmpea[:]
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
