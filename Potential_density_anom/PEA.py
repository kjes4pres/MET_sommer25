import numpy as np

def pea(rho, z_w, rhoref=1027.0, maxdepth=-10.0):

    """ 
    This function returns the potential energy anomaly, which is defined as
    the integral from a given depth below the surface to the surface, using 
    the integrand
        
        -g*max(rhoref - rho,0)*z/rhoref
                
    where rhoref is a reference density value, typically chosen as a value
    representative of open ocean conditions where there is little influence
    from riverine forcing. In reduced gravity models, the potential energy 
    anomaly can be shown to be proportional to the geostrophic transport 
    stream function, see Gustafsson (Cont. Shelf Res., 19(8), 1999).  
    
    The reference density should be such that there is some reference waters 
    in the area of interest. Keep in mind that maps based on the output from 
    this function might appear strange if the chosen maximum depth is larger 
    than the minimum depth of the model, hence it is best to mask the map where 
    the local depth is smaller than the maximum depth.
    
    Usage:
    
        potential_energy_anomaly = pea(rho, z_w, rhoref=1027.0, maxdepth=-10.0)
        
    Input variables:

        rho                         - 4D density field from ROMS (ndarray [T,Z,Y,X])
        z_w                         - 4D depth values of w-points (ndarray [T,Z,Y,X])
        rhoref                      - reference density value, default = 1027.0
        maxdepth                    - thickness of layer, default = -10.0
                                      time dependent zeta is taken into account

    Output:

        potential_energy_anomaly    -  [m^3 s^-2] (ndarray [T,Y,X])
    
    2025-04-23, kaihc@met.no
    """

    # Set constants
    g = 9.81

    # Truncate z vector, keeping in mind that the surface coordinate is time dependent
    z = np.where(z_w < maxdepth + z_w[:,-1:,:,:], maxdepth + z_w[:,-1:,:,:], z_w)

    # Calculate dz
    dz = np.diff(z, axis=1)

    # Calculate z values
    z_mid = (z[:,:-1,:,:] + z[:,1:,:,:])/2

    # Calculate integrand
    P = -g*np.max(rhoref-rho,0)*z_mid/rhoref

    # Integrate and return
    return np.sum(P*dz, axis=1)