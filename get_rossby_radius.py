'''
Module for calculating the Rossby radius of deformation.


'''

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
    #depths = np.linspace(-2500, 0) 
    depths = np.arange(-2000, -520, 20)
    depths = np.insert(depths, len(depths), values=np.arange(-300, -105, 5), axis=0)
    depths = np.insert(depths, len(depths), values=np.arange(-100, -52, 2), axis=0)
    depths = np.insert(depths, len(depths), values=np.arange(-50, 1, 1), axis=0)
    N_sliced = N.xroms.zslice(grid, depths, z=z_w)
    N_sliced = N_sliced.fillna(0)

    # Buoyancy frequency integrated over depth
    N_int = N_sliced.integrate(coord='z_w')
    
    # WKB approx. of Rossby radius
    R = (1 / (np.abs(f) * np.pi)) * N_int

    #R = R.mean(dim='ocean_time')

    return R