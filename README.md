# Workspace for summer job at the Meteorological Institute of Norway, 2025.
__Kjersti Stangeland, July 2025__

### Overview of contents

#### Rossby_deformation
* `find_deformation_radius.py`: Module for computing the baroclinic rossby radius. Code partly copied from Ann Kristin Sperrevik, which was a part of the analysis in: https://doi.org/10.1002%2F2016JC012640. Computes the deformation radius and writes to files.
* `N2.py`: Module for computing the buoyancy frequency squared. Code partly copied from Ann Kristin Sperrevik, which was a part of the analysis in: https://doi.org/10.1002%2F2016JC012640.
* `density.py`: Module for getting the density of sea water, from Bjørn Ådlandsvik <bjorn@imr.no>, 07 November 2004.
* `thought_process_rossby.ipynb`: A mess, but parts of the thought process behind. See `\possible_trash` for more.
* `funcs.py`: Module with different processing functions
* `get_turbine_coords.py`: Module for opening turbine coordinates as xArray DataSet.
* `making_study_area.ipynb`: Notebook for experimenting with defining a study area based off the internal Rossby deformation radius.
* `SV_F_rossby.ipynb`: Notebook for experimenting and investigating the internal Rossby deformation radius at Sørvest-F.


#### Mixed_layer_depth
* `mld_pd_plt.py`: Module for plotting profiles of potential density and MLD at Sørvest-F and Nordvest-C.
* `MLD.ipynb`: Notebook investigating the mixed layer depth at Sørvest-F and Nordvest-C.
* `mld.py`: Module for computing mixed layer depth using potential density threshold method.

#### Integrated potential density anomaly
*Coming*

#### Heat fluxes and surface stress
*Coming*

#### Heat and salt content
*Coming*

