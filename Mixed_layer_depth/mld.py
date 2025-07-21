'''
---Kjersti Stangeland, July 2025---

Module for reading in data from Norkyst v3 (daily files),
interpolating salt and potential temperature from s-levels to z-depths, 
calculating potentential density, mixed layer depth,
and lastly writing the radius and integrated buoyancy frequency to a netcdf file.

Code and logic strongly inspired by A. K. Sperrevik:
'/home/annks/Projects/LoVe'

Here the potential density threshold method is used to calculate MLD, with a threshold of 0.03 kgm⁻3. 

References:
Sperrevik, A. K., J. Röhrs, and K. H. Christensen (2017), 
Impact of data assimilation on Eulerian versus Lagrangian estimates of upper ocean transport, 
J. Geophys. Res. Oceans, 122, 5445–5457, doi:10.1002/2016JC012640.

Treguier, A. M., de Boyer Montégut, C., Bozec, A., Chassignet, E. P., Fox-Kemper, B., McC. Hogg, A., 
Iovino, D., Kiss, A. E., Le Sommer, J., Li, Y., Lin, P., Lique, C., Liu, H., Serazin, G., Sidorenko, D., 
Wang, Q., Xu, X., and Yeager, S.: The mixed-layer depth in the Ocean Model 
Intercomparison Project (OMIP): impact of resolving mesoscale eddies, Geosci. Model Dev., 16, 3849–3872, 
https://doi.org/10.5194/gmd-16-3849-2023, 2023.
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

    files = glob('/lustre/storeB/project/nwp/havvind/hav/results/reference//REF-06/norkyst_avg_0001.nc')
    for f in files:
        print(f)
        bvf_calc(f)

def calc_mld(pot_dens, z):
    '''
    
    '''
    # the mixed layer depth is the depth where the potential density equals the surface - a threshold
    thres = pot_den[0] - 0.03  # [kgm^⁻3]

    mld = xr.where(pot_dens > thres)

    return mld



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
    salt  = ds.variables['salt'][:]
    temp = ds.variables['temp'][:]
    f = ds.variables['f'][:]
    mask = grid.mask_rho[:,:]

    tmpm = np.ones([1, salt.shape[2], salt.shape[3]])*np.nan  # Temporary array holding mld
    
    # Making a file to write to
    outputf = '/home/kjsta7412/sommer_25/MET_sommer25/output_mld/tests/' + mfile.split('/')[-1].replace('.nc',f'__mld.nc')

    """ ref_part = mfile.split('reference//')[-1].split('/')[0].replace('-', '_')  # 'REF-02' to 'REF_02'
    norkyst_part = mfile.split('/')[-1][:-3]  # Get 'norkyst_avg_0001' without '.nc'
    filename = f"{ref_part}_{norkyst_part}_brr.nc"
    outputf = '/home/kjsta7412/sommer_25/MET_sommer25/output_bdr/REF/' + filename """

    rootgrp = Dataset(outputf, 'w')
    time =  rootgrp.createDimension("ocean_time", None)
    X = rootgrp.createDimension("xi_rho", len(ds.dimensions['xi_rho']))
    Y = rootgrp.createDimension("eta_rho", len(ds.dimensions['eta_rho']))

    otime = rootgrp.createVariable("ocean_time","f8",("ocean_time",), zlib=True)
    otime[:] = ocean_time[-1]

    mld = rootgrp.createVariable("mld","f8",("ocean_time","eta_rho","xi_rho",),  zlib=True)
    mld.long_name= "Mixed layer depth"
    mld.units = "meter"

    rootgrp.close()

    ds.close()

    # Looping through grid points
    for y in range(0, salt.shape[2]):
        for x in range(0, salt.shape[3]): 
            if not mask[y, x]:  # skipping land points
               continue

            tmpm[0,y,x] = 0    
            # Filtering out local water depth on the zlevs
            # Where zlevs is shallower than z_r -> true
            tmpmz = zlevs[np.where(zlevs[:]>z_r[:,y,x].min())].squeeze()  # z-levels to calculate potential density on

            if 1:
               t = -1
               # Adding first and last value to each end of the array to match length of z-levels (tmpmz)
               tmpmS = salt[t,:,y,x]; tmpmS = np.append(tmpmS,tmpmS[-1]); tmpmS = np.insert(tmpmS,0,values=tmpmS[0],axis=0)
               tmpmT = temp[t,:,y,x]; tmpmT = np.append(tmpmT,tmpmT[-1]); tmpmT = np.insert(tmpmT,0,values=tmpmT[0],axis=0)
               # Interpolating to z-levels
               tempZ = griddata(z_r[:,y,x], tmpmT[:], tmpmz)
               saltZ = griddata(z_r[:,y,x], tmpmS[:], tmpmz)

               # Getting the potential density 
               densm = dens(saltZ, tempZ, np.zeros_like(tempZ)).squeeze()
               
               # Caluclating mixed layer depth

               tmpm[0,y,x] = calc_mld()
               

    rootgrp = Dataset(outputf, 'r+')
    mld = rootgrp.variables['mld']
    mld[:] = tmpm[:]
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