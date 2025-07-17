'''
---Kjersti Stangeland, July 2025---

Module for reading in reference data from Norkyst v3 (daily files),
interpolating density from s-levels to z-depths, 
calculating buoyancy frequency on z-depths,
calculating Baroclinic Rossby radius (mode 1) via WKB approximation,
and lastly writing the radius and integrated buoyancy frequency to a netcdf file.

Code and logic strongly inspired by A. K. Sperrevik:
'/home/annks/Projects/LoVe'
'''

from roppy import SGrid
from netCDF4 import Dataset
import numpy as np
from scipy.interpolate import griddata
from glob import glob
from Rossby_deformation.N2 import N2
import time
from Rossby_deformation.density import dens

def main():
    base_path = glob('/lustre/storeB/project/nwp/havvind/hav/results/experiment/')
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
            file_path = f'/EXP-{month}/norkyst_avg_{day_str}.nc'
            files.append(base_path[0]+file_path)

    #files = glob('/lustre/storeB/project/nwp/havvind/hav/results/reference//REF-06/norkyst_avg_0001.nc')
    for f in files:
        print(f)
        bvf_calc(f)

def bvf_calc(mfile):
    print(mfile)
    
    ds = Dataset(mfile)
    grid = SGrid(ds) 

    # Depths to interpolate to
    # Depth levels from Annks
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
    #density = ds.variables['rho'][:] + 1000  # Full density - not just anomaly
    salt  = ds.variables['salt'][:]
    temp = ds.variables['temp'][:]
    f = ds.variables['f'][:]
    #h = grid.h[:,:]
    mask = grid.mask_rho[:,:]

    tmpn = np.ones([1, salt.shape[2], salt.shape[3]])*np.nan  # Temporary array holding integrated buoyancy frequency
    tmpr = np.ones([1, salt.shape[2], salt.shape[3]])*np.nan  # Temporary array holding Rossby radius

    # Making a file to write to
    #outputf = '/home/kjsta7412/sommer_25/MET_sommer25/output_bdr/' + mfile.split('/')[-1].replace('.nc',f'__brr.nc')

    ref_part = mfile.split('experiment//')[-1].split('/')[0].replace('-', '_')  # 'REF-02' to 'REF_02'
    norkyst_part = mfile.split('/')[-1][:-3]  # Get 'norkyst_avg_0001' without '.nc'
    filename = f"{ref_part}_{norkyst_part}_brr.nc"
    outputf = '/home/kjsta7412/sommer_25/MET_sommer25/output_bdr/EXP/' + filename

    rootgrp = Dataset(outputf, 'w')
    time =  rootgrp.createDimension("ocean_time", None)
    X = rootgrp.createDimension("xi_rho", len(ds.dimensions['xi_rho']))
    Y = rootgrp.createDimension("eta_rho", len(ds.dimensions['eta_rho']))

    #ref = datetime(1970, 1, 1)  # Reference time
    #ocean_time = np.array(ocean_time)
    #date = pd.to_datetime(ref + timedelta(seconds=ocean_time[0]))
    otime = rootgrp.createVariable("ocean_time","f8",("ocean_time",), zlib=True)
    otime[:] = ocean_time[-1]

    bvf = rootgrp.createVariable("bvf","f8",("ocean_time","eta_rho","xi_rho",),  zlib=True)
    bvf.long_name= "Brunt Vaisala Frequency"
    bvf.units = "second-1"

    gamma_r  = rootgrp.createVariable("gamma_r","f8",("ocean_time","eta_rho","xi_rho",),  zlib=True)
    gamma_r.long_name= "Baroclinic Rossby radius"
    gamma_r.units = "meter"
    rootgrp.close()

    ds.close()

    # Looping through grid points
    for y in range(0, salt.shape[2]):
        for x in range(0, salt.shape[3]): 
            if not mask[y, x]:  # skipping land points
               continue

            tmpn[0,y,x] = 0    
            # Filtering out local water depth on the zlevs
            # Where zlevs is shallower than z_r -> true
            tmpnz = zlevs[np.where(zlevs[:]>z_r[:,y,x].min())].squeeze()  # z-levels to calculate N on
            N_depth = (tmpnz[0:-1] + tmpnz[1:])/2.  # Midpoints between z-levels

            if 1:
               t = -1
               # Adding first and last value to each end of the array to match length of z-levels (tmpnz)
               tmpnS = salt[t,:,y,x]; tmpnS = np.append(tmpnS,tmpnS[-1]); tmpnS = np.insert(tmpnS,0,values=tmpnS[0],axis=0)
               tmpnT = temp[t,:,y,x]; tmpnT = np.append(tmpnT,tmpnT[-1]); tmpnT = np.insert(tmpnT,0,values=tmpnT[0],axis=0)
               # Interpolating to z-levels
               tempZ = griddata(z_r[:,y,x], tmpnT[:], tmpnz)
               saltZ = griddata(z_r[:,y,x], tmpnS[:], tmpnz)

               #tmpD =  density[t, :, y, x]
               #tmpD = np.append(tmpD, tmpD[-1]); tmpD = np.insert(tmpD, 0, values=tmpD[0], axis=0)
               #densZ = griddata(z_r[:, y, x], tmpD[:], tmpnz) 

               densm = dens(saltZ, tempZ, np.zeros_like(tempZ)).squeeze()
               # Calculating the buoyancy frequency
               n2 = N2(densm, tmpnz);  n2[np.where(n2<0)] = 0.0000001; N = np.sqrt(n2).squeeze()
               #n2 = N2(densZ, tmpnz); n2[np.where(n2<0)] = 0.0000001; N = np.sqrt(n2).squeeze()

               # Integrating N over z-levels
               for k in range(0, len(N)-1):
                   tmpn[0,y,x] = tmpn[0, y,x] + np.abs(N_depth[k] - N_depth[k+1])/2.*(N[k]+N[k+1]) 

               # Calculating the Rossby radius of deformation
               # WKB approximation 
               tmpr[0,y,x] = tmpn[0,y,x]/(f[y,x]*np.pi)
               

    rootgrp = Dataset(outputf, 'r+')
    bvf = rootgrp.variables['bvf']
    gamma_r = rootgrp.variables['gamma_r']
    bvf[:] = tmpn[:]
    gamma_r[:] = tmpr[:]
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